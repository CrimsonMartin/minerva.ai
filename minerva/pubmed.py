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
# NCBI allows 3 requests/s anonymously and 10/s with a (free) API key.
_ANON_INTERVAL = 0.35
_KEYED_INTERVAL = 0.11
_last_request = 0.0

API_KEY = ""


def configure(api_key: str = "") -> None:
    """Set the NCBI API key used by every request from here on.

    A key raises the rate limit from 3 to 10 requests/s, which matters
    because graph walks fire searches in bursts.
    """
    global API_KEY
    API_KEY = api_key or ""


def _throttle() -> None:
    global _last_request
    interval = _KEYED_INTERVAL if API_KEY else _ANON_INTERVAL
    wait = interval - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()


def _credentials(email: str) -> dict:
    """Contact address and API key, as E-utilities expects them."""
    params = {}
    if email:
        params["email"] = email
    if API_KEY:
        params["api_key"] = API_KEY
    return params


def search(term: str, retmax: int = 20, email: str = "") -> list[str]:
    """Return PMIDs for a query, sorted by relevance."""
    _throttle()
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
        **_credentials(email),
    }
    response = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=60)
    response.raise_for_status()
    return response.json()["esearchresult"].get("idlist", [])


def fetch(pmids: list[str], email: str = "") -> list[dict]:
    """Fetch paper records (title, abstract, journal, year, mesh) for PMIDs."""
    if not pmids:
        return []
    _throttle()
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
              **_credentials(email)}
    response = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=120)
    response.raise_for_status()
    return _parse_efetch(response.text)


def fetch_fulltext(pmid: str, email: str = "") -> list[dict] | None:
    """Full text paragraphs for a PMID via PubMed Central, or None.

    Only works for the PMC open-access subset: elink maps the PMID to a
    PMCID, efetch returns JATS XML, and we pull the body paragraphs with
    their section titles. Returns [{"section", "text"}, ...] or None when
    there is no PMC record or no readable body (paywalled/abstract-only).
    """
    _throttle()
    params = {"dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "json",
              **_credentials(email)}
    response = requests.get(f"{EUTILS}/elink.fcgi", params=params, timeout=60)
    response.raise_for_status()
    pmcid = None
    for linkset in response.json().get("linksets", []):
        for linksetdb in linkset.get("linksetdbs", []):
            if linksetdb.get("linkname") == "pubmed_pmc" and linksetdb.get("links"):
                pmcid = linksetdb["links"][0]
                break
    if not pmcid:
        return None

    _throttle()
    params = {"db": "pmc", "id": pmcid, "retmode": "xml", **_credentials(email)}
    response = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=120)
    response.raise_for_status()
    paragraphs = parse_jats_body(response.text)
    return paragraphs or None


def parse_jats_body(xml_text: str) -> list[dict]:
    """Extract body paragraphs (with section titles) from PMC JATS XML."""
    root = ET.fromstring(xml_text)
    body = root.find(".//body")
    if body is None:
        return []
    paragraphs: list[dict] = []

    def walk(node, section: str) -> None:
        for child in node:
            tag = child.tag.split("}")[-1]  # tolerate namespaced JATS
            if tag == "sec":
                title = child.find("title")
                name = "".join(title.itertext()).strip() if title is not None else section
                walk(child, name or section)
            elif tag == "p":
                text = " ".join("".join(child.itertext()).split())
                if text:
                    paragraphs.append({"section": section, "text": text})

    walk(body, "")
    return paragraphs


def fulltext_markdown(title: str, paragraphs: list[dict]) -> str:
    """Render fetched full text as a readable markdown document."""
    lines = [f"# {title}\n"]
    last_section = None
    for paragraph in paragraphs:
        if paragraph["section"] and paragraph["section"] != last_section:
            lines.append(f"## {paragraph['section']}\n")
            last_section = paragraph["section"]
        lines.append(paragraph["text"] + "\n")
    return "\n".join(lines)


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
