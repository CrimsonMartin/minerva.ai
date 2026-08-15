"""Agent-loop, idea-merge, document-extraction, and PubMed-parsing tests,
driven by the MockLLM. Run: python -m tests.test_agent  (from repo root)
"""

import tempfile
import zipfile
from pathlib import Path

from minerva import agent, pubmed
from minerva.config import DEFAULT_CONFIG, _merge
from minerva.embeddings import EmbeddingIndex
from minerva.ingest import extract_text
from minerva.llm import _extract_json
from minerva.mock import MockLLM
from minerva.store import Vault

FIXTURES = Path(__file__).parent / "fixtures"

# Two papers whose abstracts share one identical core-idea sentence, so the
# MockLLM (which merges exact statement matches) collapses them to one node.
SHARED = ("Loss of GPX4 activity triggers ferroptosis through unchecked lipid "
          "peroxidation in membranes of affected cells.")
PAPERS = {
    "111": {"pmid": "111", "title": "GPX4 loss and ferroptosis",
            "abstract": SHARED + " This was first shown in fibroblasts.",
            "journal": "Cell", "year": "2020", "mesh": ["Ferroptosis"]},
    "222": {"pmid": "222", "title": "GPX4 in melanoma",
            "abstract": SHARED + " Melanoma persister cells are especially sensitive.",
            "journal": "Nature", "year": "2021", "mesh": ["Melanoma"]},
}


def test_full_run_merges_shared_idea_abstract_path():
    def fake_search(term, retmax=20, email=""):
        return ["111", "222"]

    def fake_fetch(pmids, email=""):
        return [PAPERS[p] for p in pmids if p in PAPERS]

    for module in (pubmed, agent.pubmed):
        module.search, module.fetch = fake_search, fake_fetch
        module.fetch_fulltext = lambda pmid, email="": None  # force abstract path

    tmp = tempfile.mkdtemp()
    config = _merge(DEFAULT_CONFIG, {"_root": tmp, "reflect_every": 10,
                                     "llm": {"chat_model": "mock", "embed_model": "mock"},
                                     "pubmed": {"full_text": False}})
    report = agent.run_research(config, "ferroptosis", "depth", 8)

    vault = Vault(Path(tmp) / "vault")
    assert set(vault.list_papers()) == {"111", "222"}, vault.list_papers()
    # The shared sentence is one idea node carrying both papers.
    shared_nodes = [s for s in vault.list_ideas()
                    if {p["pmid"] for p in vault.load_idea(s)["papers"]} == {"111", "222"}]
    assert shared_nodes, "shared idea did not merge across the two papers"
    # Relative links resolve both directions.
    slug = shared_nodes[0]
    assert f"../../ideas/{slug}/idea.md" in (vault.papers_dir / "111" / "paper.md").read_text()
    assert "../../papers/111/paper.md" in (vault.ideas_dir / slug / "idea.md").read_text()
    assert report.exists()
    print(f"  abstract path: merged shared idea across 111+222 (slug {slug})")


def test_resume_and_new_run_selection():
    def fake_search(term, retmax=20, email=""):
        return ["111", "222"]

    def fake_fetch(pmids, email=""):
        return [PAPERS[p] for p in pmids if p in PAPERS]

    for module in (pubmed, agent.pubmed):
        module.search, module.fetch = fake_search, fake_fetch
        module.fetch_fulltext = lambda pmid, email="": None

    tmp = tempfile.mkdtemp()
    config = _merge(DEFAULT_CONFIG, {"_root": tmp, "reflect_every": 10,
                                     "llm": {"chat_model": "mock", "embed_model": "mock"},
                                     "pubmed": {"full_text": False}})
    first = agent.run_research(config, "ferroptosis", "depth", 3).parent
    # --resume lands in the same run dir and its state carries over.
    resumed = agent.run_research(config, "ferroptosis", "depth", 3, resume=True).parent
    assert resumed == first, (resumed, first)
    # --new refuses to reuse the duplicate topic's dir and gets a -2 suffix.
    fresh = agent.run_research(config, "ferroptosis", "depth", 3, resume=False).parent
    assert fresh != first and fresh.name == f"{first.name}-2", fresh.name
    # A later --resume picks the most recent matching run (the -2 one).
    latest = agent._latest_matching_run(first.parent, "ferroptosis", "depth")
    assert latest == fresh, (latest, fresh)
    print(f"  run selection: resume reuses {first.name}, new forks {fresh.name}")


def test_breadth_vs_depth_scoring():
    from minerva.frontier import Frontier
    depth = Frontier(Path(tempfile.mktemp()), alpha=1.0, beta=0.15)
    breadth = Frontier(Path(tempfile.mktemp()), alpha=0.55, beta=1.0)
    depth.domain_counts["oncology"] = 5
    breadth.domain_counts["oncology"] = 5
    # A highly relevant but familiar-domain item: depth ranks it far above breadth.
    d = depth.score(relevance=0.9, domains=["oncology"])
    b = breadth.score(relevance=0.9, domains=["oncology"])
    assert d > b, (d, b)
    # A novel-domain item: breadth rewards it much more than depth does.
    d_novel = depth.score(relevance=0.4, domains=["botany"])
    b_novel = breadth.score(relevance=0.4, domains=["botany"])
    assert (b_novel - b) > (d_novel - d), "breadth should reward novelty more than depth"
    print(f"  scoring: depth favors familiar-relevant, breadth favors novel domains")


def test_docx_and_extraction():
    tmp = Path(tempfile.mkdtemp())
    docx = tmp / "m.docx"
    xml = ('<?xml version="1.0"?><w:document '
           'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           '<w:body><w:p><w:r><w:t>GPX4 loss triggers ferroptosis.</w:t></w:r></w:p>'
           '<w:p><w:r><w:t>Iron is required.</w:t></w:r></w:p></w:body></w:document>')
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("word/document.xml", xml)
    text = extract_text(docx)
    assert "GPX4 loss triggers ferroptosis." in text and "Iron is required." in text
    txt = tmp / "n.txt"
    txt.write_text("plain text note")
    assert extract_text(txt) == "plain text note"
    print("  extraction: docx + txt OK")


def test_efetch_xml_parsing():
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation>
<PMID>999</PMID><Article>
<Journal><Title>Test J</Title><JournalIssue><PubDate><Year>2019</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>A <i>title</i> here</ArticleTitle>
<Abstract><AbstractText Label="AIM">Do things.</AbstractText><AbstractText>More.</AbstractText></Abstract>
</Article><MeshHeadingList><MeshHeading><DescriptorName>Mice</DescriptorName></MeshHeading></MeshHeadingList>
</MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    parsed = pubmed._parse_efetch(xml)[0]
    assert parsed["title"] == "A title here"
    assert parsed["abstract"] == "AIM: Do things.\nMore."
    assert parsed["mesh"] == ["Mice"] and parsed["year"] == "2019"
    print("  efetch XML parsing OK")


def test_json_repair():
    assert _extract_json('noise {"a": {"b": "}"}} trailing')["a"]["b"] == "}"
    print("  JSON repair extraction OK")


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        print(f"- {test.__name__}")
        test()
    print(f"\nALL {len(tests)} AGENT TESTS PASSED")


if __name__ == "__main__":
    _run()
