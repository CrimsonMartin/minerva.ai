"""Final synthesis: notebook + idea network -> a cited markdown report."""

from pathlib import Path

from .extract import _prompt
from .llm import LLM, LLMError
from .notebook import Notebook
from .pdf import render_pdf
from .store import Vault


# How many ideas the synthesis sees per paper of research done, and the
# floor below which a report has too little network context to be useful.
IDEAS_PER_PAPER = 10
MIN_IDEAS = 40
MAX_IDEA_CHARS = 60_000   # keep the prompt inside a local model's context


def synthesize(llm: LLM, vault: Vault, notebook: Notebook, run_dir: Path,
               topic: str, mode: str, log=None, budget: int = 0) -> Path:
    log = log or (lambda message: None)
    path = run_dir / "report.md"
    # Scale the network context with the size of the research: a 30-paper run
    # earns a richer report than a 3-paper one. Findings count papers already
    # read, so a synthesis-only resume (--budget 0) keeps the scale it earned.
    effort = max(budget, len(notebook.findings))
    limit = max(MIN_IDEAS, effort * IDEAS_PER_PAPER)
    ideas, shown = _idea_summary(vault, limit=limit)
    log(f"synthesizing report from {len(notebook.findings)} finding(s) and "
        f"{shown} of {len(vault.list_ideas())} idea(s) — one long call, "
        f"may take minutes")
    user = (
        f"Topic: {topic}\nMode: {mode}\n\n"
        f"Findings notebook:\n{notebook.as_text() or '(empty)'}\n\n"
        f"Idea network (statement · type · domain · paper count · relations):\n{ideas}"
    )
    try:
        # The synthesis is one big call where reasoning quality matters most —
        # it may use a different model or request options than the small calls.
        report = llm.chat(_prompt("synthesize"), user,
                          model=getattr(llm, "report_model", None),
                          extra_body=getattr(llm, "report_extra_body", None),
                          timeout=getattr(llm, "report_timeout", None))
        if not report.strip():
            # A reasoning model that burns its whole token budget thinking
            # returns empty content — that's a failure, not a report.
            raise LLMError("synthesis returned no content (max_tokens too "
                           "small for a reasoning model?)")
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
    path.write_text(report if report.endswith("\n") else report + "\n")
    try:  # a bad PDF render must never lose the markdown report
        pdf = render_pdf(path)
    except Exception as exc:
        log(f"report PDF failed (report.md is intact): {exc}")
    else:
        log(f"report PDF written: {pdf}" if pdf
            else "report PDF skipped — `pip install markdown-pdf` to enable")
    return path


def _idea_summary(vault: Vault, limit: int) -> tuple[str, int]:
    """The most-cited ideas as one line each. Returns (text, how many)."""
    rows = []
    for slug in vault.list_ideas():
        idea = vault.load_idea(slug)
        rows.append(
            (
                len(idea["papers"]),
                f"- {idea['statement']} · {idea['type']} · {idea['domain']} · "
                f"{len(idea['papers'])} papers · "
                + (", ".join(f"{e['relation']}:{e['target']}" for e in idea["edges"][:4])
                   or "no relations"),
            )
        )
    rows.sort(key=lambda row: row[0], reverse=True)
    lines, size = [], 0
    for _, text in rows[:limit]:
        if size + len(text) > MAX_IDEA_CHARS:  # context guard, whatever the limit
            break
        lines.append(text)
        size += len(text) + 1
    return ("\n".join(lines) or "(no ideas extracted)"), len(lines)
