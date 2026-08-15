"""Final synthesis: notebook + idea network -> a cited markdown report."""

from pathlib import Path

from .extract import _prompt
from .llm import LLM, LLMError
from .notebook import Notebook
from .store import Vault


def synthesize(llm: LLM, vault: Vault, notebook: Notebook, run_dir: Path,
               topic: str, mode: str, log=None) -> Path:
    log = log or (lambda message: None)
    path = run_dir / "report.md"
    ideas = _idea_summary(vault, limit=40)
    log(f"synthesizing report from {len(notebook.findings)} finding(s) and "
        f"{len(vault.list_ideas())} idea(s) — one long call, may take minutes")
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
                          extra_body=getattr(llm, "report_extra_body", None))
        if not report.strip():
            # A reasoning model that burns its whole token budget thinking
            # returns empty content — that's a failure, not a report.
            raise LLMError("synthesis returned no content (max_tokens too "
                           "small for a reasoning model?)")
    except Exception as exc:  # never lose a run's findings to a bad last call
        report = (
            f"# {topic} — synthesis failed\n\n"
            f"The synthesis call failed ({exc}). The raw findings are in "
            f"notebook.md and the idea network is in the vault's ideas/ folder.\n"
        )
    path.write_text(report if report.endswith("\n") else report + "\n")
    return path


def _idea_summary(vault: Vault, limit: int) -> str:
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
    return "\n".join(text for _, text in rows[:limit]) or "(no ideas extracted)"
