# minerva.ai

A local-first **deep research agent** over PubMed. It decomposes papers
into their **core ideas** (building blocks), links those ideas across
articles into a network, and explores a topic with a tunable knob:

- **depth** — drill into exactly how one idea works, level by level
- **breadth** — hunt for the same idea used creatively across distant fields

Everything runs on your machine: a local LLM served by **LM Studio**
(or any OpenAI-compatible endpoint — vLLM, llama.cpp, Ollama) and a
**files-only backend**.
No database. The knowledge graph is a folder of `.json` + `.md` files you
can read, grep, and open in Obsidian (graph view draws your idea network).

## Long papers: recursive paper trees

Papers of any length are handled by building a **tree** bottom-up:
paragraphs are the leaves; consecutive leaves are grouped and each group
is condensed into a summary + ideas by one LLM call; those summaries are
grouped and condensed again; and so on up to a single root — the paper's
"topic level". Every call sees a bounded amount of text (config
`tree.group_chars`), so context never blows up no matter how long the
paper is, and the number of calls grows linearly with length.

Ideas are canonicalized **top-down** (root first), so a new paper enters
the graph at the topic level and its finer ideas anchor beneath. Child
ideas get a `part_of` edge to their parent node's idea, projecting the
paper's own structure into the idea network. Each paper folder gets a
`tree.json` (source of truth) and a readable `tree.md` outline.

**Cross-level linking.** Every idea records the tree `level` it came from
(0 = a specific leaf claim, higher = broader/topic level). When a new idea
is canonicalized it is embedded and compared against *all* existing ideas
regardless of level, through two gates: candidates above `merge_threshold`
may be merged (same idea), while candidates above the lower `link_threshold`
are offered as link-only. That lower gate is what lets a specific claim in
one paper reach a broader, differently-worded idea in another — cross-level
pairs never clear the high merge bar, so without it they'd never be seen.
The model links them with `part_of` (or `analogous_to`, etc.), and the edge
is always oriented specific → broad by comparing levels, whichever paper
introduced which.

Full text comes from **PubMed Central** when a paper is open-access
(`elink` → PMCID → JATS XML → section-aware paragraphs); otherwise the
agent falls back to the abstract. Local files (`--input`) go through the
same tree builder. Toggle full text with `pubmed.full_text` in config.

## Vaults — the folder structure is the network

A vault is one independent knowledge graph. You can keep several —
one per research programme — each in its own subdirectory of `vaults/`:

```
vaults/<name>/
  papers/<pmid>/paper.json    one paper (source of truth)
  papers/<pmid>/paper.md      readable view → links to its core ideas
  ideas/<slug>/idea.json      one canonical idea across ALL papers
  ideas/<slug>/idea.md        readable view → links to papers + related ideas
  index/embeddings.json       the only "index": id → embedding vector
  runs/<date-topic-mode>/     one research session
    run.json  frontier.json   config + resumable explore queue
    notebook.md  log.md       findings (cited) + live activity log
    report.md  report.pdf     the final synthesized, cited report
                              (.pdf only when markdown-pdf is installed)
```

Every command works in one vault: `--vault NAME` (or `MINERVA_VAULT` in
the environment/`.env`) picks it, `main` is the default, and a vault is
created on first use. `minerva vaults` lists them with their sizes.
Papers, ideas, embeddings, and runs never cross vault boundaries — which
also means the same idea in two vaults is two separate nodes.
A pre-multi-vault layout (a single top-level `vault/` folder) is adopted
automatically: it is moved to `vaults/main` the first time it's needed.

Ideas are typed (`mechanism`, `method`, `finding`, `problem`) and linked
by typed edges (`part_of`, `causes`, `enables`, `analogous_to`, …).
`analogous_to` is the interesting one: same trick, different domain —
found by "embedding similar + domain different", which is breadth mode's
whole game.

## How a run works

A deterministic **frontier loop** owns control flow; the LLM only does
small, single-purpose calls (extract ideas from one abstract, adjudicate
one merge, reflect on the notebook). That keeps a local model coherent
over a 50–200 step run.

Every structured call carries a **JSON schema** (`response_format:
json_schema`), so a capable server — LM Studio, llama.cpp, vLLM —
constrains decoding to the exact shape the code expects and the reply
can't be malformed or missing a key. Servers without it degrade
gracefully: `json_schema` → `json_object` → a prompt-only JSON reply
parsed with one repair retry.

1. Seed: PubMed search for your topic → papers onto the frontier.
2. Pop the best item, forever:
   - **paper** → fetch + cache → extract core ideas → merge into the
     vault (embedding kNN + LLM adjudication) → note the key finding
   - **idea** → follow graph edges, link cross-domain analogies, spawn
     new PubMed queries
   - **query** → search PubMed, score results, push papers
3. Scoring: `score = α·relevance + β·domain_novelty` — depth is high α,
   breadth is high β. That one line is the depth/breadth knob.
4. Every 8 steps: a reflection call reviews the notebook, names gaps,
   injects new queries, or declares the question answered.
5. Synthesis: notebook + idea network → `report.md` with PMID citations.

Extraction is **lazy**: only papers the agent actually touches get the
LLM treatment, so the graph grows organically around your topics and
every run makes future runs on nearby topics warmer.

## Setup

```bash
uv venv && uv pip install -r requirements.txt   # or plain pip install -r
```

Two model roles, one required decision:

- **Chat** — any OpenAI-compatible server (LM Studio, llama.cpp, vLLM,
  Ollama), local or on another box. Copy `.env.example` to `.env` and set
  `MINERVA_LLM_BASE_URL` + `MINERVA_LLM_CHAT_MODEL` to whatever your
  server reports.
- **Embeddings** — none needed. They run in-process by default
  (Qwen3-Embedding-0.6B via sentence-transformers, auto-downloaded on
  first use, fine on CPU). To use a server instead, set
  `MINERVA_LLM_EMBED_MODEL` (and optionally
  `MINERVA_LLM_EMBED_BASE_URL` when it's a different endpoint than chat).

Real environment variables win over `.env`. Everything else —
thresholds, scoring weights, tree sizes — lives in `DEFAULT_CONFIG` in
`minerva/config.py`. The vault is created on first use.

One caveat: embedding vectors are only comparable within one model, so
if you ever change the embed model, start a fresh vault.

## Use

```bash
python -m minerva research "ferroptosis in cancer therapy" --mode depth --budget 50
python -m minerva research "predator-prey oscillation models" --mode breadth --budget 80
python -m minerva --vault ecology research "predator-prey oscillation models" \
                                       # same command, separate knowledge graph
python -m minerva vaults               # list vaults with paper/idea counts
python -m minerva ideas                # list the idea network by paper count
python -m minerva graph                # render the network as an interactive
                                       # self-contained <vault>/graph.html
```

### PDF reports

Runs always end in a markdown `report.md` — that stays the source of
truth (greppable, linkable, Obsidian-friendly). If the optional
`markdown-pdf` package is installed, every synthesis *also* writes a
`report.pdf` next to it, and `minerva pdf FILE...` renders any markdown
file (an old report, a paper.md) on demand:

```bash
pip install markdown-pdf               # one package: markdown-it-py + PyMuPDF,
                                       # plain wheels, no system dependencies
python -m minerva pdf vaults/main/runs/<run>/report.md
```

Without it, PDF rendering is skipped with a hint and nothing else changes.

The graph page searches by **meaning**, with no server involved: rendering
re-embeds every idea with a small sentence model (all-MiniLM-L6-v2) and
ships those vectors in the file, and the page embeds your query with the
same model in the browser via WebGPU (transformers.js, weights fetched and
cached on first search). So "tiny particles that deliver drugs" finds the
nanomedicine ideas with no shared wording. Offline or on an old browser it
falls back to plain text matching, and the box says which mode it is in.
Each idea's panel also lists its nearest *unlinked* ideas — that one is
pure arithmetic on the shipped vectors and always works.

The search model is deliberately *not* the vault's embedding model: the
vault's model drives idea canonicalization and should be the best one you
can serve, while the page needs one small enough to run in a browser. They
never interact — the page's vectors and its query come from the same small
model, so they are consistent with each other.

### Resuming vs. duplicating a run

Every run lives in its own `<vault>/runs/<date-topic-mode>/` directory, and
everything a run needs to continue (frontier queue, notebook, log) reloads
from that directory. Two flags control which directory a `research`
command uses:

```bash
python -m minerva research "ferroptosis" --resume   # continue the most recent
                                                    # run for this topic+mode
python -m minerva research "ferroptosis" --new      # fresh run, even though the
                                                    # topic duplicates an old one
```

`--resume` picks up the latest matching run wherever it left off — visited
items stay visited, findings keep accumulating, and the report is
re-synthesized from the combined notebook. `--new` starts from scratch in a
fresh directory (suffixed `-2`, `-3`, … when needed) while still sharing
the vault's idea network. Without either flag, rerunning the same topic on
the same day implicitly resumes it; on a later day it starts a new run.

### Drilling into an existing idea

`--resume` continues a question; `--seed-idea` asks a *new* question that
starts exploring from a chosen node of the shared graph. The new topic
re-centers the frontier's relevance scoring on the sub-idea (resuming
keeps the old topic's center of gravity), and the run gets its own
notebook and report — while everything it finds still canonicalizes into
the same vault:

```bash
python -m minerva research "RSL3 inhibition of GPX4" --mode depth \
    --seed-idea rsl3-covalently-inhibits-the-gpx4-selenocysteine-1a2b3c
```

The slug is the idea's folder name under `<vault>/ideas/` (also shown by
`minerva ideas` and the graph page's detail panel).

### Local files as the research base

Hand the agent your own documents — a manuscript draft, a review PDF,
notes — and it decomposes them into ideas first, links them into the
network as `local-<hash>` papers, and explores outward from *their*
ideas before anything the topic search finds:

```bash
python -m minerva research "ferroptosis" --mode depth --input draft.docx --input review.pdf
python -m minerva ingest paper1.pdf paper2.docx    # add to the vault without a run
```

Supported: `.pdf`, `.docx`, `.txt`, `.md`. Digital PDFs are read via
their text layer (pypdf). Scanned PDFs fall back to **PaddleOCR** —
that's optional and only needed for scans:

```bash
pip install paddlepaddle paddleocr
```

Ingest is chunked (config `ingest.chunk_chars` / `max_chunks`) so long
documents stay within a local model's comfortable context, and the same
file bytes are never processed twice.

Watch a run live: `tail -f vaults/main/runs/<run>/log.md` — or just open
`notebook.md` in your editor and watch findings accumulate.

### Sandboxed (bubblewrap)

```bash
./run_sandboxed.sh research "..." --mode depth
```

Read-only system, writable only this project dir, shared network (needed
for localhost LM Studio + PubMed).

## Testing / dry-run

Set `MINERVA_LLM_CHAT_MODEL=mock` (in the environment or `.env`) to run
the entire pipeline — tree building, ingestion, a full research session —
with a deterministic mock model and no network or GPU. The offline test
suite uses it:

```bash
python -m tests.run_all
```

## License

MIT — see [LICENSE](LICENSE). Use it, fork it, build on it.
