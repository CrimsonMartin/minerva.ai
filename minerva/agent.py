"""The orchestrator: a deterministic frontier loop around small LLM calls.

The loop — not the model — owns control flow. Each step pops the best
frontier item and does one bounded unit of work with a fresh, small
context: read a paper, expand an idea, or run a query. Every `reflect_every`
steps a planning call reviews the notebook and injects new queries.
This is what keeps a local model coherent over a long research run.
"""

import datetime
import json
import re
from pathlib import Path

from . import pubmed
from .config import vault_path
from .embeddings import EmbeddingIndex, cosine
from .extract import canonicalize, extract_ideas, _prompt
from .frontier import Frontier
from .llm import LLMError
from .mock import make_llm
from .notebook import Notebook
from .report import synthesize
from .store import Vault, slugify

REFLECT_SCHEMA = {
    "title": "reflection",
    "type": "object",
    "properties": {
        "assessment": {"type": "string"},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "queries": {"type": "array", "items": {"type": "string"}},
        "done": {"type": "boolean"},
    },
    "required": ["assessment", "gaps", "queries", "done"],
    "additionalProperties": False,
}


def run_research(config: dict, topic: str, mode: str, budget: int,
                 inputs: list[Path] | None = None,
                 resume: bool | None = None,
                 seed_ideas: list[str] | None = None) -> Path:
    """resume=True continues the latest run for this topic+mode, resume=False
    forces a fresh run dir even when one exists for the topic, and None keeps
    the default (a same-day rerun of a topic implicitly resumes it)."""
    llm = make_llm(config)
    vault = Vault(vault_path(config))
    index = EmbeddingIndex(vault.root, model=config["llm"]["embed_model"])
    email = config["pubmed"]["email"]

    run_dir, note = _select_run_dir(vault.runs_dir, topic, mode, resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps({"topic": topic, "mode": mode, "budget": budget}, indent=2) + "\n"
    )

    weights = config["scoring"][mode]
    frontier = Frontier(run_dir / "frontier.json", weights["alpha"], weights["beta"])
    notebook = Notebook(run_dir, topic, mode)

    topic_vector = llm.embed(topic)
    index.add(f"topic:{run_dir.name}", topic_vector)
    index.save()

    log = _make_logger(run_dir)
    log(f"run start · topic={topic!r} mode={mode} budget={budget}")
    if note:
        log(note)

    for path in inputs or []:
        _seed_from_input(config, llm, vault, index, frontier, notebook,
                         Path(path), topic_vector, log)

    for slug in seed_ideas or []:
        if not vault.has_idea(slug):
            log(f"seed idea not in vault, skipping: {slug}")
            continue
        idea = vault.load_idea(slug)
        log(f"seed idea: {idea['statement'][:80]} [{slug}]")
        # The user chose this node as the starting point — explore it before
        # anything the topic search finds (same precedence as --input ideas).
        frontier.push("idea", slug, 1.0, [idea["domain"]], "seed idea")

    if not frontier.visited:  # fresh run (not a resume) — seed from the topic too
        _seed(frontier, topic, config["pubmed"]["seed_results"], email, log)

    # The budget counts papers actually read (LLM extraction work) — the
    # expensive unit. Graph walks and query searches are near-free and don't
    # consume it; a pop cap keeps them from looping unbounded on a warm vault.
    steps = 0
    pops = 0
    pop_cap = budget * 25
    last_reflect = 0
    while steps < budget and pops < pop_cap:
        item = frontier.pop()
        if item is None:
            log("frontier empty — reflecting for new directions")
            if not _reflect(llm, frontier, notebook, topic, mode, log):
                break
            continue
        pops += 1
        log(f"read {steps}/{budget} · {item['kind']} {item['id']} "
            f"(score {item['score']}) — {item['reason']}")
        try:
            if item["kind"] == "query":
                _do_query(frontier, vault, index, llm, item["id"], topic_vector,
                          config["pubmed"]["expand_results"], email)
            elif item["kind"] == "paper":
                if _do_paper(frontier, vault, index, llm, notebook, item["id"],
                             topic_vector, config, email, log):
                    steps += 1
            elif item["kind"] == "idea":
                _do_idea(frontier, vault, index, llm, item["id"], topic_vector, mode)
        except (LLMError, pubmed.requests.RequestException) as exc:
            log(f"  step failed, continuing: {exc}")
        if steps - last_reflect >= config["reflect_every"]:
            last_reflect = steps
            if _reflect(llm, frontier, notebook, topic, mode, log):
                pass
            else:
                log("reflection says done")
                break

    log(f"run end · {steps} paper(s) read over {pops} frontier steps, "
        f"{len(notebook.findings)} findings, {len(vault.list_ideas())} ideas in vault")
    report_path = synthesize(llm, vault, notebook, run_dir, topic, mode, log=log,
                             budget=budget, index=index)
    log(f"report written: {report_path}")
    return report_path


# ---------------------------------------------------------- run selection

def _select_run_dir(runs_dir: Path, topic: str, mode: str,
                    resume: bool | None) -> tuple[Path, str]:
    """Pick the run dir for this session, plus a log note about the choice.

    Everything a run needs to continue (frontier, notebook, log) lives in
    the dir and reloads from disk, so resuming is just pointing here at an
    existing one.
    """
    stamp = datetime.date.today().strftime("%Y%m%d")
    default = runs_dir / f"{stamp}-{slugify(topic)[:40]}-{mode}"
    if resume is True:
        latest = _latest_matching_run(runs_dir, topic, mode)
        if latest is not None:
            return latest, f"resuming earlier run {latest.name}"
        return default, "no earlier run for this topic+mode — starting fresh"
    if resume is False:
        candidate, n = default, 2
        while candidate.exists():
            candidate = default.with_name(f"{default.name}-{n}")
            n += 1
        if candidate != default:
            return candidate, f"{default.name} exists — starting duplicate run {candidate.name}"
        return candidate, ""
    return default, ""


def _latest_matching_run(runs_dir: Path, topic: str, mode: str) -> Path | None:
    """Most recent run dir whose recorded topic+mode match (by topic slug)."""
    slug = slugify(topic)[:40]
    matches = []
    for run_dir in runs_dir.iterdir() if runs_dir.exists() else []:
        run_json = run_dir / "run.json"
        if not run_json.exists():
            continue
        try:
            recorded = json.loads(run_json.read_text())
        except (ValueError, OSError):
            continue
        if recorded.get("mode") == mode and slugify(recorded.get("topic", ""))[:40] == slug:
            matches.append(run_dir)
    # Names start with a YYYYMMDD stamp, so lexicographic order is chronological.
    return max(matches, key=lambda p: p.name) if matches else None


# ------------------------------------------------------------------ steps

def _seed_from_input(config, llm, vault, index, frontier, notebook,
                     path: Path, topic_vector, log) -> None:
    """Ingest a local pdf/docx/txt file and seed the frontier with its ideas."""
    from .ingest import IngestError, ingest_file

    try:
        doc = ingest_file(
            llm, vault, index, path, config["merge_threshold"],
            tree_config=config["tree"], link_threshold=config.get("link_threshold"),
            min_chars_per_page=config["ingest"]["min_chars_per_page"],
            log=log,
        )
    except IngestError as exc:
        log(f"input skipped: {exc}")
        return
    cached = " (already in vault)" if doc["cached"] else ""
    log(f"input {path.name} → {doc['id']}{cached}, {len(doc['slugs'])} ideas")
    if doc.get("tree") and doc["tree"]["truncated"]:
        log(f"  WARNING: {doc['tree']['truncated']} paragraphs beyond "
            f"tree.max_paragraphs were dropped")
    for finding in doc["findings"]:
        notebook.note(finding, [doc["id"]], via=path.name)
    for slug in doc["slugs"]:
        idea = vault.load_idea(slug)
        vector = index.get(f"idea:{slug}")
        relevance = cosine(vector, topic_vector) if vector else 0.5
        # The user handed us this document as the research base — its ideas
        # should be explored before anything a seed query turns up.
        frontier.push("idea", slug, max(relevance, 0.9), [idea["domain"]],
                      f"research base: {path.name}")


def _seed(frontier: Frontier, topic: str, retmax: int, email: str, log) -> None:
    pmids = pubmed.search(topic, retmax=retmax, email=email)
    log(f"seed query returned {len(pmids)} papers")
    for rank, pmid in enumerate(pmids):
        relevance = 1.0 - rank / max(len(pmids), 1) * 0.3
        frontier.push("paper", pmid, relevance, [], f"seed result #{rank + 1}")


def _do_query(frontier, vault, index, llm, query, topic_vector, retmax, email) -> None:
    for pmid in pubmed.search(query, retmax=retmax, email=email):
        relevance, domains = _paper_signals(vault, index, llm, topic_vector, pmid, email)
        frontier.push("paper", pmid, relevance, domains, f"from query: {query}")


def _do_paper(frontier, vault, index, llm, notebook, pmid, topic_vector,
              config, email, log) -> bool:
    """Returns True when the paper was actually read (extraction work ran) —
    the unit the run budget counts. Cached papers cost nothing."""
    read = False
    if not vault.has_paper(pmid):
        fetched = pubmed.fetch([pmid], email=email)
        if not fetched:
            return False
        vault.save_paper(fetched[0])
    paper = vault.load_paper(pmid)
    if not paper.get("ideas"):  # lazy extraction: only papers the agent touches
        if not _read_fulltext(vault, index, llm, notebook, paper, config, email, log):
            if not paper.get("abstract"):
                return False
            extracted = extract_ideas(llm, paper)
            canonicalize(llm, vault, index, pmid, extracted, config["merge_threshold"],
                         link_threshold=config.get("link_threshold"), level=0)
            if extracted["key_finding"]:
                notebook.note(extracted["key_finding"], [pmid])
        read = True
        paper = vault.load_paper(pmid)
    for link in paper["ideas"]:
        idea = vault.load_idea(link["slug"])
        vector = index.get(f"idea:{link['slug']}")
        relevance = cosine(vector, topic_vector) if vector else 0.5
        frontier.push("idea", link["slug"], relevance, [idea["domain"]],
                      f"idea from PMID {pmid}")
    return read


def _read_fulltext(vault, index, llm, notebook, paper, config, email, log) -> bool:
    """Try the PMC full-text + recursive-tree path. True when it handled the paper."""
    if not config["pubmed"].get("full_text", True) or paper.get("source"):
        return False
    pmid = paper["pmid"]
    try:
        paragraphs = pubmed.fetch_fulltext(pmid, email=email)
    except pubmed.requests.RequestException as exc:
        log(f"  full text fetch failed for {pmid}, using abstract: {exc}")
        return False
    if not paragraphs or len(paragraphs) < 3:
        return False  # not open access (or trivially short) — abstract path

    from .tree import build_paper_tree, split_paragraphs

    (vault.papers_dir / pmid / "fulltext.md").write_text(
        pubmed.fulltext_markdown(paper["title"], paragraphs)
    )
    text = "\n\n".join(p["text"] for p in paragraphs)
    settings = config["tree"]
    leaves = split_paragraphs(text, settings["leaf_chars"])
    log(f"  full text: {len(leaves)} leaves — building tree")
    result = build_paper_tree(
        llm, vault, index, pmid, leaves, config["merge_threshold"],
        link_threshold=config.get("link_threshold"),
        group_chars=settings["group_chars"], max_paragraphs=settings["max_paragraphs"],
        log=log,
    )
    log(f"  full text: {result['n_leaves']} leaves → tree depth {result['depth']}, "
        f"{len(result['slugs'])} ideas")
    if result["truncated"]:
        log(f"  WARNING: {result['truncated']} paragraphs beyond "
            f"tree.max_paragraphs were dropped")
    paper = vault.load_paper(pmid)
    paper["summary"] = result["summary"]
    vault.save_paper(paper)
    if result["key_finding"]:
        notebook.note(result["key_finding"], [pmid])
    return True


def _do_idea(frontier, vault, index, llm, slug, topic_vector, mode) -> None:
    idea = vault.load_idea(slug)
    vector = index.get(f"idea:{slug}")

    # Follow existing graph edges.
    for edge in idea["edges"]:
        if vault.has_idea(edge["target"]):
            other = vault.load_idea(edge["target"])
            other_vector = index.get(f"idea:{edge['target']}")
            relevance = cosine(other_vector, topic_vector) if other_vector else 0.5
            frontier.push("idea", edge["target"], relevance, [other["domain"]],
                          f"{edge['relation']} of {slug}")

    # Link analogies already in the vault: similar vector, different domain.
    if vector:
        for key, score in index.search(vector, k=8, prefix="idea:", exclude=f"idea:{slug}"):
            other_slug = key.removeprefix("idea:")
            if not vault.has_idea(other_slug):
                continue
            other = vault.load_idea(other_slug)
            if score >= 0.72 and other["domain"] != idea["domain"]:
                vault.link_ideas(slug, other_slug, "analogous_to",
                                 f"similarity {score:.2f}, cross-domain")
                relevance = cosine(index.get(key), topic_vector)
                frontier.push("idea", other_slug, relevance, [other["domain"]],
                              f"analogy of {slug}")

    # Turn the idea into a new literature query.
    query = _idea_query(idea, mode)
    frontier.push("query", query, 0.8 if mode == "depth" else 0.6, [idea["domain"]],
                  f"expand idea {slug}")


def _idea_query(idea: dict, mode: str) -> str:
    entities = idea.get("entities", [])[:3]
    if mode == "depth" and entities:
        return " AND ".join(entities) + " mechanism"
    if entities:
        return " OR ".join(entities)
    words = re.findall(r"[A-Za-z][A-Za-z-]{3,}", idea["statement"])
    return " ".join(words[:6])


def _paper_signals(vault, index, llm, topic_vector, pmid, email):
    """Relevance + domains for a paper, fetching and embedding it if needed."""
    if not vault.has_paper(pmid):
        fetched = pubmed.fetch([pmid], email=email)
        if not fetched:
            return 0.0, []
        vault.save_paper(fetched[0])
    paper = vault.load_paper(pmid)
    key = f"paper:{pmid}"
    vector = index.get(key)
    if vector is None and paper.get("abstract"):
        vector = llm.embed(f"{paper['title']}\n{paper['abstract'][:1500]}")
        index.add(key, vector)
        index.save()
    relevance = cosine(vector, topic_vector) if vector else 0.3
    return relevance, paper.get("mesh", [])[:5]


def _reflect(llm, frontier, notebook, topic, mode, log) -> bool:
    """Planning call. Returns False when the run should stop."""
    user = (
        f"Topic: {topic}\nMode: {mode}\n\nNotebook so far:\n"
        f"{notebook.as_text() or '(empty)'}"
    )
    try:
        result = llm.chat_json(_prompt("reflect"), user, REFLECT_SCHEMA)
    except LLMError as exc:
        log(f"  reflect failed, continuing: {exc}")
        return True
    if result.get("assessment"):
        log(f"  reflect: {result['assessment'][:200]}")
    queries = [q for q in result.get("queries", []) if isinstance(q, str) and q.strip()]
    for query in queries[:3]:
        frontier.push("query", query.strip(), 0.9, [], "reflection gap")
    if result.get("done") is True:
        return False
    return bool(queries) or len(frontier) > 0


def _make_logger(run_dir: Path):
    log_path = run_dir / "log.md"

    def log(message: str) -> None:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"- `{stamp}` {message}"
        print(message, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    return log
