"""Command line interface.

    python -m minerva init
    python -m minerva research "ferroptosis in cancer therapy" --mode depth --budget 50
    python -m minerva research "attention mechanisms" --mode breadth
    python -m minerva ideas
"""

import argparse
import sys

from .config import load_config, vault_path, write_default_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="minerva", description="Local-first deep research agent over PubMed."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="write minerva.config.json and create the vault")

    research = sub.add_parser("research", help="run a research session")
    research.add_argument("topic", help="the research question or topic")
    research.add_argument("--mode", choices=["depth", "breadth"], default="depth",
                          help="depth: drill into the mechanism; "
                               "breadth: hunt for the idea across domains")
    research.add_argument("--budget", type=int, default=50,
                          help="frontier steps (bounds LLM calls and wall time)")
    research.add_argument("--input", action="append", default=[], metavar="FILE",
                          help="local .pdf/.docx/.txt/.md to ingest as the research "
                               "base (repeatable); its ideas seed the frontier")

    ingest = sub.add_parser("ingest", help="ingest local files into the vault "
                                           "without running a research session")
    ingest.add_argument("files", nargs="+", help=".pdf/.docx/.txt/.md paths")

    sub.add_parser("ideas", help="list ideas in the vault by paper count")

    args = parser.parse_args(argv)
    config = load_config()

    if args.command == "init":
        path = write_default_config()
        from .store import Vault
        Vault(vault_path(config))
        print(f"config: {path}\nvault:  {vault_path(config)}")
        return 0

    if args.command == "research":
        from pathlib import Path

        from .agent import run_research
        report = run_research(config, args.topic, args.mode, args.budget,
                              inputs=[Path(p) for p in args.input])
        print(f"\nreport: {report}")
        return 0

    if args.command == "ingest":
        from pathlib import Path

        from .embeddings import EmbeddingIndex
        from .ingest import IngestError, ingest_file
        from .mock import make_llm
        from .store import Vault
        llm = make_llm(config)
        vault = Vault(vault_path(config))
        index = EmbeddingIndex(vault.root)
        failures = 0
        for file in args.files:
            try:
                doc = ingest_file(
                    llm, vault, index, Path(file), config["merge_threshold"],
                    tree_config=config["tree"], link_threshold=config.get("link_threshold"),
                    min_chars_per_page=config["ingest"]["min_chars_per_page"],
                )
            except IngestError as exc:
                print(f"{file}: FAILED — {exc}")
                failures += 1
                continue
            cached = " (already in vault)" if doc["cached"] else ""
            tree = doc.get("tree")
            shape = (f", tree depth {tree['depth']} over {tree['n_leaves']} paragraphs"
                     if tree else "")
            print(f"{file}: {doc['id']}{cached}, {len(doc['slugs'])} ideas{shape}")
        return 1 if failures else 0

    if args.command == "ideas":
        from .store import Vault
        vault = Vault(vault_path(config))
        rows = []
        for slug in vault.list_ideas():
            idea = vault.load_idea(slug)
            rows.append((len(idea["papers"]), idea["statement"], slug))
        rows.sort(reverse=True)
        for count, statement, slug in rows:
            print(f"{count:4d}  {statement}  [{slug}]")
        if not rows:
            print("(vault has no ideas yet — run a research session)")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
