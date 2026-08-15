"""PubMed access via NCBI E-utilities, with a files-only cache.

Instead of indexing the 36M-abstract baseline locally, we fetch on
demand and cache every paper we ever touch under vault/papers/<pmid>/.
The vault becomes a growing personal library of everything the agent
has read.
"""

import time
import xml.etree.ElementTree as ET

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_MIN_INTERVAL = 0.35  # stay under NCBI's 3 req/s unauthenticated limit
_last_request = 0.0


def _throttle() -> None:
    global _last_request
    wait = _MIN_INTERVAL - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()


def search(term: str, retmax: int = 20, email: str = "") -> list[str]:
    """Return PMIDs for a query, sorted by relevance."""
    _throttle()
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
    }
    if email:
        params["email"] = email
    response = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=60)
    response.raise_for_status()
    return response.json()["esearchresult"].get("idlist", [])


def fetch(pmids: list[str], email: str = "") -> list[dict]:
    """Fetch paper records (title, abstract, journal, year, mesh) for PMIDs."""
    if not pmids:
        return []
    _throttle()
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    if email:
        params["email"] = email
    response = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=120)
    response.raise_for_status()
    return _parse_efetch(response.text)


def _parse_efetch(xml_text: str) -> list[dict]:
    papers = []
    root = ET.fromstring(xml_text)
    for article in root.findall(".//PubmedArticle"):
        citation = article.find("MedlineCitation")
        if citation is None:
            continue
        pmid = citation.findtext("PMID", "")
        art = citation.find("Article")
        if art is None or not pmid:
            continue
        abstract_parts = []
        for node in art.findall(".//AbstractText"):
            label = node.get("Label")
            text = "".join(node.itertext()).strip()
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
        mesh = [
            node.findtext("DescriptorName", "").strip()
            for node in citation.findall(".//MeshHeading")
        ]
        papers.append(
            {
                "pmid": pmid,
                "title": "".join(art.find("ArticleTitle").itertext()).strip()
                if art.find("ArticleTitle") is not None
                else "",
                "abstract": "\n".join(abstract_parts),
                "journal": art.findtext("Journal/Title", ""),
                "year": art.findtext("Journal/JournalIssue/PubDate/Year", ""),
                "mesh": [term for term in mesh if term],
            }
        )
    return papers
