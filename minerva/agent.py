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
from .llm import LLM, LLMError
from .notebook import Notebook
from .report import synthesize
from .store import Vault, slugify


def run_research(config: dict, topic: str, mode: str, budget: int,
                 inputs: list[Path] | None = None) -> Path:
    llm = LLM(config)
    vault = Vault(vault_path(config))
    index = EmbeddingIndex(vault.root)
    email = config["pubmed"]["email"]

    stamp = datetime.date.today().strftime("%Y%m%d")
    run_dir = vault.runs_dir / f"{stamp}-{slugify(topic)[:40]}-{mode}"
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

    for path in inputs or []:
        _seed_from_input(config, llm, vault, index, frontier, notebook,
                         Path(path), topic_vector, log)

    if not frontier.visited:  # fresh run (not a resume) — seed from the topic too
        _seed(frontier, topic, config["pubmed"]["seed_results"], email, log)

    steps = 0
    while steps < budget:
        item = frontier.pop()
        if item is None:
            log("frontier empty — reflecting for new directions")
            if not _reflect(llm, frontier, notebook, topic, mode, log):
                break
            continue
        steps += 1
        log(f"step {steps}/{budget} · {item['kind']} {item['id']} "
            f"(score {item['score']}) — {item['reason']}")
        try:
            if item["kind"] == "query":
                _do_query(frontier, vault, index, llm, item["id"], topic_vector,
                          config["pubmed"]["expand_results"], email)
            elif item["kind"] == "paper":
                _do_paper(frontier, vault, index, llm, notebook, item["id"],
                          topic_vector, config["merge_threshold"], email)
            elif item["kind"] == "idea":
                _do_idea(frontier, vault, index, llm, item["id"], topic_vector, mode)
        except (LLMError, pubmed.requests.RequestException) as exc:
            log(f"  step failed, continuing: {exc}")
        if steps % config["reflect_every"] == 0:
            if _reflect(llm, frontier, notebook, topic, mode, log):
                pass
            else:
                log("reflection says done")
                break

    log(f"run end · {steps} steps, {len(notebook.findings)} findings, "
        f"{len(vault.list_ideas())} ideas in vault")
    report_path = synthesize(llm, vault, notebook, run_dir, topic, mode)
    log(f"report written: {report_path}")
    return report_path


# ------------------------------------------------------------------ steps

def _seed_from_input(config, llm, vault, index, frontier, notebook,
                     path: Path, topic_vector, log) -> None:
    """Ingest a local pdf/docx/txt file and seed the frontier with its ideas."""
    from .ingest import IngestError, ingest_file

    settings = config["ingest"]
    try:
        doc = ingest_file(
            llm, vault, index, path, config["merge_threshold"],
            chunk_chars=settings["chunk_chars"], max_chunks=settings["max_chunks"],
            min_chars_per_page=settings["min_chars_per_page"],
        )
    except IngestError as exc:
        log(f"input skipped: {exc}")
        return
    cached = " (already in vault)" if doc["cached"] else ""
    log(f"input {path.name} → {doc['id']}{cached}, {len(doc['slugs'])} ideas")
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
              merge_threshold, email) -> None:
    if not vault.has_paper(pmid):
        fetched = pubmed.fetch([pmid], email=email)
        if not fetched:
            return
        vault.save_paper(fetched[0])
    paper = vault.load_paper(pmid)
    if not paper.get("abstract"):
        return
    if not paper.get("ideas"):  # lazy extraction: only papers the agent touches
        extracted = extract_ideas(llm, paper)
        canonicalize(llm, vault, index, pmid, extracted, merge_threshold)
        if extracted["key_finding"]:
            notebook.note(extracted["key_finding"], [pmid])
        paper = vault.load_paper(pmid)
    for link in paper["ideas"]:
        idea = vault.load_idea(link["slug"])
        vector = index.get(f"idea:{link['slug']}")
        relevance = cosine(vector, topic_vector) if vector else 0.5
        frontier.push("idea", link["slug"], relevance, [idea["domain"]],
                      f"idea from PMID {pmid}")


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
        result = llm.chat_json(_prompt("reflect"), user)
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
