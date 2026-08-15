# minerva.ai

A local-first **deep research agent** over PubMed. It decomposes papers
into their **core ideas** (building blocks), links those ideas across
articles into a network, and explores a topic with a tunable knob:

- **depth** — drill into exactly how one idea works, level by level
- **breadth** — hunt for the same idea used creatively across distant fields

Everything runs on your machine: a local LLM (Qwen via Ollama/vLLM/
llama.cpp — any OpenAI-compatible endpoint) and a **files-only backend**.
No database. The knowledge graph is a folder of `.json` + `.md` files you
can read, grep, and open in Obsidian (graph view draws your idea network).

## The vault — the folder structure is the network

```
vault/
  papers/<pmid>/paper.json    one paper (source of truth)
  papers/<pmid>/paper.md      readable view → links to its core ideas
  ideas/<slug>/idea.json      one canonical idea across ALL papers
  ideas/<slug>/idea.md        readable view → links to papers + related ideas
  index/embeddings.json       the only "index": id → embedding vector
  runs/<date-topic-mode>/     one research session
    run.json  frontier.json   config + resumable explore queue
    notebook.md  log.md       findings (cited) + live activity log
    report.md                 the final synthesized, cited report
```

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
pip install -r requirements.txt        # just: requests (flask etc. for the old web UI)
ollama pull qwen3:14b                  # or any chat model you like
ollama pull nomic-embed-text           # embedding model
python -m minerva init                 # writes minerva.config.json + vault/
```

Edit `minerva.config.json` to point at your endpoint/models.

## Use

```bash
python -m minerva research "ferroptosis in cancer therapy" --mode depth --budget 50
python -m minerva research "predator-prey oscillation models" --mode breadth --budget 80
python -m minerva ideas                # list the idea network by paper count
```

Watch a run live: `tail -f vault/runs/<run>/log.md` — or just open
`notebook.md` in your editor and watch findings accumulate.

### Sandboxed (bubblewrap)

```bash
./run_sandboxed.sh research "..." --mode depth
```

Read-only system, writable only this project dir, shared network (needed
for localhost Qwen + PubMed).

## Legacy

`app.py`, `createAndUploadToIndex.py` etc. are the earlier Flask +
Elasticsearch experiment; the agent does not depend on them.
