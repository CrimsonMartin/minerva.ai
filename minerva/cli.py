"""Command line interface.

    python -m minerva research "ferroptosis in cancer therapy" --mode depth --budget 50
    python -m minerva research "attention mechanisms" --mode breadth
    python -m minerva ideas
"""

import argparse
import sys

from .config import load_config, vault_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="minerva", description="Local-first deep research agent over PubMed."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    research = sub.add_parser("research", help="run a research session")
    research.add_argument("topic", help="the research question or topic")
    research.add_argument("--mode", choices=["depth", "breadth"], default="depth",
                          help="depth: drill into the mechanism; "
                               "breadth: hunt for the idea across domains")
    research.add_argument("--budget", type=int, default=50,
                          help="papers to read (bounds LLM extraction work; "
                               "graph walks and query searches are free)")
    research.add_argument("--input", action="append", default=[], metavar="FILE",
                          help="local .pdf/.docx/.txt/.md to ingest as the research "
                               "base (repeatable); its ideas seed the frontier")
    research.add_argument("--seed-idea", action="append", default=[], metavar="SLUG",
                          help="existing vault idea to start exploring from "
                               "(repeatable); explored before anything the topic "
                               "search finds")
    fresh_or_resume = research.add_mutually_exclusive_group()
    fresh_or_resume.add_argument("--resume", action="store_true",
                                 help="continue the most recent run for this topic and "
                                      "mode: its frontier, notebook, and findings pick "
                                      "up where they left off")
    fresh_or_resume.add_argument("--new", action="store_true",
                                 help="always start a fresh run, even when this topic "
                                      "already has a run directory (a duplicate)")

    ingest = sub.add_parser("ingest", help="ingest local files into the vault "
                                           "without running a research session")
    ingest.add_argument("files", nargs="+", help=".pdf/.docx/.txt/.md paths")

    sub.add_parser("ideas", help="list ideas in the vault by paper count")

    graph = sub.add_parser("graph", help="render the idea network as a "
                                         "self-contained interactive HTML file")
    graph.add_argument("-o", "--out", default="vault/graph.html",
                       help="output path (default: vault/graph.html)")
    graph.add_argument("--title", default="Idea Network",
                       help="page title (default: Idea Network)")
    graph.add_argument("--no-embed-search", action="store_true",
                       help="omit the embedding endpoint from the page; its "
                            "search box then matches text only (the endpoint "
                            "URL is otherwise baked in, so use this for a page "
                            "you intend to share)")

    args = parser.parse_args(argv)
    config = load_config()

    if args.command == "research":
        from pathlib import Path

        from .agent import run_research
        resume = True if args.resume else False if args.new else None
        report = run_research(config, args.topic, args.mode, args.budget,
                              inputs=[Path(p) for p in args.input], resume=resume,
                              seed_ideas=args.seed_idea)
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
        index = EmbeddingIndex(vault.root, model=config["llm"]["embed_model"])
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

    if args.command == "graph":
        from pathlib import Path

        from .graph_html import render_graph_html
        from .store import Vault
        vault = Vault(vault_path(config))
        if not vault.list_ideas():
            print("(vault has no ideas yet — run a research session first)")
            return 1
        embed_search = None
        if not args.no_embed_search and not config["llm"]["embed_model"].startswith("local:"):
            embed_search = {
                "url": (config["llm"].get("embed_base_url")
                        or config["llm"]["base_url"]).rstrip("/") + "/embeddings",
                "model": config["llm"]["embed_model"],
                "apiKey": config["llm"]["api_key"],
            }
        path = render_graph_html(vault, Path(args.out), title=args.title,
                                 embed_search=embed_search)
        print(f"graph written: {path}")
        if embed_search:
            print(f"  semantic search via {embed_search['url']} "
                  f"({embed_search['model']}), falling back to text matching")
        return 0

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
