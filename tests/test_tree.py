"""End-to-end tests for the recursive paper tree, PMC parsing, and a full
research run — all driven by the deterministic MockLLM, no network, no
real model. Run: python -m tests.test_tree   (from the repo root)
"""

import json
import tempfile
from pathlib import Path

from minerva import pubmed
from minerva.config import DEFAULT_CONFIG, _merge
from minerva.embeddings import EmbeddingIndex
from minerva.ingest import ingest_file
from minerva.mock import MockLLM
from minerva.store import Vault
from minerva.tree import build_paper_tree, split_paragraphs

FIXTURES = Path(__file__).parent / "fixtures"


def _fresh_vault():
    root = Path(tempfile.mkdtemp())
    return Vault(root), EmbeddingIndex(root)


def test_split_paragraphs_handles_any_length():
    # A 10k-char single paragraph with no breaks must still become leaves.
    leaves = split_paragraphs("word " * 2000, leaf_chars=1200)
    assert len(leaves) > 5
    assert all(len(leaf) <= 1200 for leaf in leaves)
    # Real multi-paragraph text keeps paragraph grouping.
    leaves = split_paragraphs((FIXTURES / "long_paper.txt").read_text(), leaf_chars=1200)
    assert all(len(leaf) <= 1200 for leaf in leaves)
    print(f"  split_paragraphs: long paper -> {len(leaves)} leaves")


def test_tree_builds_recursively_and_bounds_calls():
    vault, index = _fresh_vault()
    llm = CountingMock()
    paragraphs = split_paragraphs((FIXTURES / "long_paper.txt").read_text(), leaf_chars=1200)
    vault.save_paper({"pmid": "P1", "title": "Ferroptosis review",
                      "abstract": "", "journal": "", "year": "", "mesh": []})
    result = build_paper_tree(llm, vault, index, "P1", paragraphs, 0.8,
                              group_chars=1500, max_paragraphs=500)

    assert result["depth"] >= 2, f"expected a multi-level tree, got depth {result['depth']}"
    assert result["n_leaves"] == len(paragraphs)
    assert result["summary"], "root must have a summary"
    assert result["slugs"], "tree must yield ideas"
    # Every summarize call saw a bounded amount of content.
    assert llm.max_content <= 1500 + 500, f"a call saw {llm.max_content} chars (unbounded!)"
    # Calls are sub-linear in a useful sense: far fewer than one-per-char.
    print(f"  tree: {len(paragraphs)} leaves -> depth {result['depth']}, "
          f"{result['n_nodes']} nodes, {llm.calls} summarize calls, "
          f"largest call {llm.max_content} chars")

    tree = json.loads((vault.papers_dir / "P1" / "tree.json").read_text())
    root = tree["nodes"][tree["root"]]
    assert root["level"] == result["depth"]
    assert (vault.papers_dir / "P1" / "tree.md").exists()


def test_part_of_edges_project_structure():
    vault, index = _fresh_vault()
    paragraphs = split_paragraphs((FIXTURES / "long_paper.txt").read_text())
    vault.save_paper({"pmid": "P2", "title": "t", "abstract": "",
                      "journal": "", "year": "", "mesh": []})
    build_paper_tree(MockLLM(), vault, index, "P2", paragraphs, 0.8, group_chars=1500)
    # At least one idea should be part_of another (child section under parent).
    part_of = 0
    for slug in vault.list_ideas():
        for edge in vault.load_idea(slug)["edges"]:
            if edge["relation"] == "part_of":
                part_of += 1
    assert part_of > 0, "expected part_of edges linking child ideas to parents"
    print(f"  structure projection: {part_of} part_of edges across "
          f"{len(vault.list_ideas())} ideas")


def test_jats_parsing_real_format():
    xml = (FIXTURES / "sample_jats.xml").read_text()
    paragraphs = pubmed.parse_jats_body(xml)
    assert len(paragraphs) >= 6, paragraphs
    sections = {p["section"] for p in paragraphs}
    assert "Introduction" in sections and "Iron dependence" in sections, sections
    assert any("selenocysteine" in p["text"] for p in paragraphs)
    md = pubmed.fulltext_markdown("GPX4 review", paragraphs)
    assert "## Introduction" in md and "## Iron dependence" in md
    print(f"  JATS: {len(paragraphs)} paragraphs across {len(sections)} sections")


def test_ingest_long_file_via_tree():
    vault, index = _fresh_vault()
    doc = ingest_file(MockLLM(), vault, index, FIXTURES / "long_paper.txt", 0.8,
                      tree_config={"leaf_chars": 1200, "group_chars": 1500,
                                   "max_paragraphs": 500})
    assert doc["tree"]["depth"] >= 2
    assert doc["summary"]
    assert (vault.papers_dir / doc["id"] / "tree.md").exists()
    assert (vault.papers_dir / doc["id"] / "fulltext.md").exists()
    paper_md = (vault.papers_dir / doc["id"] / "paper.md").read_text()
    assert "[tree](tree.md)" in paper_md and "## Summary" in paper_md
    # Re-ingest is cached.
    again = ingest_file(MockLLM(), vault, index, FIXTURES / "long_paper.txt", 0.8)
    assert again["cached"]
    print(f"  ingest: depth {doc['tree']['depth']}, {len(doc['slugs'])} root-first ideas")


def test_full_run_with_pmc_fulltext():
    """A full research run where PubMed is stubbed to return the JATS fixture."""
    jats = (FIXTURES / "sample_jats.xml").read_text()

    def fake_search(term, retmax=20, email=""):
        return ["555"]

    def fake_fetch(pmids, email=""):
        return [{"pmid": p, "title": "GPX4 and ferroptosis", "abstract": "short abstract",
                 "journal": "J", "year": "2022", "mesh": ["Ferroptosis"]} for p in pmids]

    def fake_fulltext(pmid, email=""):
        return pubmed.parse_jats_body(jats)

    from minerva import agent
    for module in (pubmed, agent.pubmed):
        module.search, module.fetch, module.fetch_fulltext = (
            fake_search, fake_fetch, fake_fulltext)

    tmp = tempfile.mkdtemp()
    config = _merge(DEFAULT_CONFIG, {"_root": tmp, "reflect_every": 3,
                                     "llm": {"chat_model": "mock", "embed_model": "mock"}})
    report = agent.run_research(config, "ferroptosis in cancer therapy", "depth", 6)

    vault = Path(tmp) / "vault"
    assert (vault / "papers" / "555" / "tree.json").exists(), "full-text tree not built"
    assert (vault / "papers" / "555" / "fulltext.md").exists()
    ideas = list((vault / "ideas").iterdir())
    assert ideas, "run produced no ideas"
    assert report.exists() and "report" in report.read_text().lower()
    print(f"  full run: built tree for PMID 555, {len(ideas)} ideas in vault")


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        print(f"- {test.__name__}")
        test()
    print(f"\nALL {len(tests)} TREE TESTS PASSED")


class CountingMock(MockLLM):
    """MockLLM that records how many summarize calls ran and the largest input."""
    calls = 0
    max_content = 0

    def chat(self, system, user, json_mode=False):
        if "condensing one node" in system:
            self.calls += 1
            self.max_content = max(self.max_content, len(user))
        return super().chat(system, user, json_mode)


if __name__ == "__main__":
    _run()
