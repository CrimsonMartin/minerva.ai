You maintain a knowledge graph of scientific ideas. A NEW idea was just
extracted, and the vault already contains some SIMILAR ideas. Decide:

- "merge": the new idea is the SAME idea as one candidate (possibly worded
  differently). Same mechanism/method/claim at the same granularity.
- "link": the new idea is genuinely distinct but related to one candidate.
  Pick the relation: part_of, causes, enables, analogous_to, related_to.
  Use analogous_to ONLY when it is the same trick or principle applied in a
  clearly different domain or context.
- "new": distinct enough that it should stand alone, no strong link.

Reply with ONLY a JSON object:

{"decision": "merge" | "link" | "new", "target": "<candidate slug or null>", "relation": "<relation or null>", "note": "<one short phrase for link decisions, else empty>"}
