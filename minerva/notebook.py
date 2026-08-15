"""The research notebook: findings as they accumulate, with citations.

notebook.json is the source of truth; notebook.md is the live, readable
view you can watch grow during a run.
"""

import json
from pathlib import Path


class Notebook:
    def __init__(self, run_dir: Path, topic: str, mode: str):
        self.json_path = run_dir / "notebook.json"
        self.md_path = run_dir / "notebook.md"
        self.topic = topic
        self.mode = mode
        self.findings: list[dict] = []
        if self.json_path.exists():
            self.findings = json.loads(self.json_path.read_text())

    def note(self, finding: str, pmids: list[str], via: str = "") -> None:
        if not finding:
            return
        self.findings.append({"finding": finding, "pmids": pmids, "via": via})
        self.save()

    def save(self) -> None:
        self.json_path.write_text(json.dumps(self.findings, indent=2) + "\n")
        lines = [f"# Notebook — {self.topic}\n", f"*mode: {self.mode}*\n"]
        for entry in self.findings:
            cites = " ".join(f"[PMID {p}]" for p in entry["pmids"])
            via = f" *(via {entry['via']})*" if entry.get("via") else ""
            lines.append(f"- {entry['finding']} {cites}{via}")
        self.md_path.write_text("\n".join(lines) + "\n")

    def as_text(self, limit: int = 120) -> str:
        recent = self.findings[-limit:]
        return "\n".join(
            f"- {entry['finding']} ({', '.join('PMID ' + p for p in entry['pmids'])})"
            for entry in recent
        )
