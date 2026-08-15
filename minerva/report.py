"""Final synthesis: notebook + idea network -> a cited markdown report."""

from pathlib import Path

from .extract import _prompt
from .llm import LLM
from .notebook import Notebook
from .store import Vault


def synthesize(llm: LLM, vault: Vault, notebook: Notebook, run_dir: Path,
               topic: str, mode: str) -> Path:
    path = run_dir / "report.md"
    ideas = _idea_summary(vault, limit=40)
    user = (
        f"Topic: {topic}\nMode: {mode}\n\n"
        f"Findings notebook:\n{notebook.as_text() or '(empty)'}\n\n"
        f"Idea network (statement · type · domain · paper count · relations):\n{ideas}"
    )
    try:
        report = llm.chat(_prompt("synthesize"), user)
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
