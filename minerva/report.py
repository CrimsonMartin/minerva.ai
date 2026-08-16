"""Final synthesis: notebook + idea network -> a cited markdown report.

The report is built the way a paper tree is built, but in reverse. A tree
condenses a document bottom-up into one root; synthesis expands one topic
top-down into its parts: the topic is split into the questions it asks,
each question is written as its own section from the ideas nearest to it,
and a last call writes the summary over the finished sections.

Every call therefore sees one question's worth of material instead of the
whole vault, which is what keeps a local model specific — the same reason
the tree exists.
"""

from pathlib import Path

from .embeddings import EmbeddingIndex, cosine
from .extract import _prompt
from .llm import LLM, LLMError
from .notebook import Notebook
from .pdf import render_pdf
from .store import Vault

QUESTIONS_SCHEMA = {
    "title": "questions",
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
    "additionalProperties": False,
}

IDEAS_PER_SECTION = 40   # network context for one question, not the whole vault
MAX_QUESTIONS = 5


def synthesize(llm: LLM, vault: Vault, notebook: Notebook, run_dir: Path,
               topic: str, mode: str, log=None, budget: int = 0,
               index: EmbeddingIndex | None = None) -> Path:
    log = log or (lambda message: None)
    path = run_dir / "report.md"
    findings = notebook.as_text() or "(empty)"

    try:
        questions = _decompose(llm, topic, log)
        log(f"synthesizing {len(questions)} section(s) from "
            f"{len(notebook.findings)} finding(s) and "
            f"{len(vault.list_ideas())} idea(s)")
        sections = []
        for i, question in enumerate(questions, 1):
            ideas = _ideas_for(llm, vault, index, question)
            log(f"  section {i}/{len(questions)}: {question[:70]}")
            body = _call(llm, "section", (
                f"Question: {question}\nMode: {mode}\n\n"
                f"Findings notebook:\n{findings}\n\n"
                f"Ideas closest to this question "
                f"(statement · type · domain · papers):\n{ideas}"
            ))
            sections.append((question, body.strip()))

        log("  summary and open questions")
        joined = "\n\n".join(f"## {q}\n\n{b}" for q, b in sections)
        head = _call(llm, "summarize_report",
                     f"Topic: {topic}\n\nSections:\n\n{joined}").strip()
        report = _assemble(topic, head, sections)
    except Exception as exc:  # never lose a run's findings to a bad last call
        note = (
            f"# {topic} — synthesis failed\n\n"
            f"The synthesis call failed ({exc}). The raw findings are in "
            f"notebook.md and the idea network is in the vault's ideas/ folder.\n"
        )
        # A failed re-synthesis must not destroy the report a previous run
        # produced; leave it in place and record the failure beside it.
        if path.exists():
            (run_dir / "report.failed.md").write_text(note)
            log(f"synthesis failed, keeping the existing report: {exc}")
            return path
        path.write_text(note)
        return path

    path.write_text(report)
    try:  # a bad PDF render must never lose the markdown report
        pdf = render_pdf(path)
    except Exception as exc:
        log(f"report PDF failed (report.md is intact): {exc}")
    else:
        log(f"report PDF written: {pdf}" if pdf
            else "report PDF skipped — `pip install markdown-pdf` to enable")
    return path


def _call(llm: LLM, prompt: str, user: str) -> str:
    """One synthesis call, with the report's model/options/timeout."""
    return llm.chat(_prompt(prompt), user,
                    model=getattr(llm, "report_model", None),
                    extra_body=getattr(llm, "report_extra_body", None),
                    timeout=getattr(llm, "report_timeout", None))


def _decompose(llm: LLM, topic: str, log) -> list[str]:
    """The distinct questions the topic asks (the topic itself as fallback)."""
    try:
        result = llm.chat_json(_prompt("decompose"), f"Topic: {topic}",
                               QUESTIONS_SCHEMA)
        questions = [q.strip() for q in result.get("questions", [])
                     if isinstance(q, str) and q.strip()]
    except (LLMError, ValueError) as exc:
        log(f"  topic decomposition failed, writing one section: {exc}")
        questions = []
    return questions[:MAX_QUESTIONS] or [topic]


def _ideas_for(llm: LLM, vault: Vault, index: EmbeddingIndex | None,
               question: str) -> str:
    """The vault's ideas nearest this question, as one line each.

    With an index this is a real retrieval — each section gets the part of
    the network that bears on it. Without one, fall back to the most-cited
    ideas so the report still has context.
    """
    slugs = []
    if index is not None:
        try:
            vector = llm.embed(question)
            slugs = [key.removeprefix("idea:") for key, _ in
                     index.search(vector, k=IDEAS_PER_SECTION, prefix="idea:")]
        except Exception:
            slugs = []
    if not slugs:
        ranked = sorted(vault.list_ideas(),
                        key=lambda s: len(vault.load_idea(s)["papers"]),
                        reverse=True)
        slugs = ranked[:IDEAS_PER_SECTION]
    lines = []
    for slug in slugs:
        if not vault.has_idea(slug):
            continue
        idea = vault.load_idea(slug)
        lines.append(f"- {idea['statement']} · {idea['type']} · "
                     f"{idea['domain']} · {len(idea['papers'])} papers")
    return "\n".join(lines) or "(no ideas extracted)"


def _assemble(topic: str, head: str, sections: list[tuple[str, str]]) -> str:
    """Stitch the parts into the report; sources are computed, not written."""
    import re

    title = topic[:1].upper() + topic[1:]
    body = [f"## {question}\n\n{text}" for question, text in sections]
    # The summary call emits "## Open questions"; those belong at the end.
    head_text, _, open_questions = head.partition("## Open questions")
    parts = [f"# {title}", "## Executive summary", head_text.strip()]
    parts += body
    if open_questions.strip():
        parts += ["## Open questions", open_questions.strip()]
    text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(p for p in parts if p.strip()))
    pmids = sorted(set(re.findall(r"PMID (\d+)", text)), key=int)
    if pmids:
        text += "\n\n## Sources\n\n" + "\n".join(f"PMID {p}" for p in pmids) + "\n"
    return text if text.endswith("\n") else text + "\n"
