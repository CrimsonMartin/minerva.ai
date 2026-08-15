"""Cross-level idea merging: a specific claim in one paper must connect to a
broader idea in another paper by embedding similarity, even though they are
worded differently and sit at different tree levels.

Uses the real canonicalize() + merge adjudication path, but overrides only the
embedding so similarities are exact and the level logic is what's under test.
Run: python -m tests.test_crosslevel  (from the repo root)
"""

import math
import tempfile
from pathlib import Path

from minerva.embeddings import EmbeddingIndex
from minerva.extract import canonicalize
from minerva.mock import MockLLM
from minerva.store import Vault

MERGE, LINK = 0.80, 0.62

BROAD = "GPX4 protects cells from ferroptotic death"
SPECIFIC = "RSL3 covalently inhibits the GPX4 selenocysteine active site"
UNRELATED = "Photosynthesis fixes atmospheric carbon in chloroplasts"


def _unit(theta: float) -> list[float]:
    return [math.cos(theta), math.sin(theta)]


class ControlledLLM(MockLLM):
    """MockLLM with a fixed embedding map so cosine similarities are exact."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed(self, text: str) -> list[float]:
        if text in self.vectors:
            return self.vectors[text]
        return super().embed(text)  # deterministic fallback for anything else


def _vault():
    root = Path(tempfile.mkdtemp())
    return Vault(root), EmbeddingIndex(root)


def _add(llm, vault, index, pmid, statement, level, kind="mechanism"):
    if not vault.has_paper(pmid):
        vault.save_paper({"pmid": pmid, "title": f"paper {pmid}", "abstract": "",
                          "journal": "", "year": "", "mesh": []})
    return canonicalize(
        llm, vault, index, pmid,
        {"key_finding": "", "ideas": [{
            "statement": statement, "type": kind, "relation_to_paper": "supports",
            "domain": "biology", "entities": []}]},
        MERGE, link_threshold=LINK, level=level,
    )


def test_specific_links_up_to_broad_across_papers():
    # cosine(broad, specific) = 0.70 → between LINK (0.62) and MERGE (0.80).
    sim = 0.70
    llm = ControlledLLM({
        BROAD: _unit(0.0),
        SPECIFIC: _unit(math.acos(sim)),
        UNRELATED: _unit(math.pi / 2),  # orthogonal to both
    })
    vault, index = _vault()

    # Paper A contributes the BROAD idea at a high tree level.
    (broad_slug,) = _add(llm, vault, index, "A", BROAD, level=2)
    # Paper B contributes the SPECIFIC idea at a leaf level.
    (specific_slug,) = _add(llm, vault, index, "B", SPECIFIC, level=0)

    assert specific_slug != broad_slug, "distinct ideas must not merge"
    specific = vault.load_idea(specific_slug)
    # The specific idea is part_of the broad one (small → large direction).
    part_of = [e for e in specific["edges"]
               if e["relation"] == "part_of" and e["target"] == broad_slug]
    assert part_of, f"expected specific part_of broad; edges={specific['edges']}"
    print(f"  cross-level: '{SPECIFIC[:30]}…' (L0) --part_of--> "
          f"'{BROAD[:30]}…' (L2) at sim {sim}")


def test_direction_is_by_level_not_insertion_order():
    """Even if the BROAD idea is added second, part_of still points small→large."""
    sim = 0.70
    llm = ControlledLLM({BROAD: _unit(0.0), SPECIFIC: _unit(math.acos(sim))})
    vault, index = _vault()
    (specific_slug,) = _add(llm, vault, index, "B", SPECIFIC, level=0)
    (broad_slug,) = _add(llm, vault, index, "A", BROAD, level=2)

    # The edge lives on the specific (lower-level) idea pointing up to broad.
    specific = vault.load_idea(specific_slug)
    assert any(e["relation"] == "part_of" and e["target"] == broad_slug
               for e in specific["edges"]), specific["edges"]
    # And broad carries only the mirrored back-edge, not a part_of of its own.
    broad = vault.load_idea(broad_slug)
    assert not any(e["relation"] == "part_of" for e in broad["edges"]), broad["edges"]
    print("  direction: broad-added-second still yields specific --part_of--> broad")


def test_below_link_threshold_stays_separate():
    # cosine = 0.40 < LINK (0.62): too far to even link → two islands.
    llm = ControlledLLM({BROAD: _unit(0.0), UNRELATED: _unit(math.acos(0.40))})
    vault, index = _vault()
    (broad_slug,) = _add(llm, vault, index, "A", BROAD, level=2)
    (other_slug,) = _add(llm, vault, index, "B", UNRELATED, level=0)
    assert other_slug != broad_slug
    assert not vault.load_idea(other_slug)["edges"], "should be no cross-link below threshold"
    assert not vault.load_idea(broad_slug)["edges"]
    print("  threshold: sim 0.40 < 0.62 leaves ideas unlinked, as intended")


def test_high_similarity_same_level_still_merges():
    # Identical statement, same level, similarity 1.0 → merge into one node.
    llm = ControlledLLM({BROAD: _unit(0.0)})
    vault, index = _vault()
    (first,) = _add(llm, vault, index, "A", BROAD, level=1)
    (second,) = _add(llm, vault, index, "B", BROAD, level=1)
    assert first == second, "identical ideas must merge to one node"
    assert {p["pmid"] for p in vault.load_idea(first)["papers"]} == {"A", "B"}
    print("  merge: identical idea across papers A+B collapses to one node")


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        print(f"- {test.__name__}")
        test()
    print(f"\nALL {len(tests)} CROSS-LEVEL TESTS PASSED")


if __name__ == "__main__":
    _run()
