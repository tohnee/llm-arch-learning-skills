---
name: llm-architecture-tracker
description: >-
  Continuously research, analyze, and report on the world's leading large
  language models — their architecture, capabilities, and pricing. Use this
  skill whenever the user wants to track, compare, survey, or deep-dive into
  frontier LLMs (OpenAI/GPT, Anthropic/Claude, Google/Gemini, DeepSeek, GLM/Z.ai,
  Kimi/Moonshot, Xiaomi MiMo, Qwen, StepFun, Mistral, Llama, NVIDIA Nemotron,
  MiniMax, and others). Trigger it for requests like "update me on the latest
  models", "compare model X and Y architectures", "generate an LLM landscape
  report", "what changed in DeepSeek V4", "build me a model comparison", "track
  new releases", "what are the top models on OpenRouter", or any request that
  involves collecting model technical reports / HuggingFace cards / ModelScope /
  OpenRouter usage + pricing and turning them into a rigorous architecture-and-
  capability analysis. Also use it to produce the recurring HTML research
  report. Use this even when the user only names one model or asks a seemingly
  simple "what's new" question — the value is in the systematic, sourced,
  architecture-level analysis this skill enforces.
license: For internal research and reporting use.
---

# LLM Architecture & Capability Tracker

This skill turns Claude into a disciplined analyst that tracks the frontier LLM
landscape end-to-end: it collects primary sources, extracts architecture and
capability details rigorously, gathers live pricing and usage data, and emits a
polished, diagram-rich HTML research report on a recurring basis.

The guiding principle is **primary sources over summaries, structure over
vibes**. A benchmark number means little without knowing the architecture,
active-parameter count, attention mechanism, context strategy, and price that
produced it. This skill always reconstructs that full picture.

## Non-negotiable rules (read first — these prevent the most common failures)

These five rules are hard requirements. A report that violates any of them is
defective and must be fixed before delivery.

1. **Output language follows the user.** If the user writes in Chinese (or asks
   for a 中文版/中文报告), the *entire* HTML report — headings, prose, table
   headers, diagram labels, captions — must be in Simplified Chinese. Keep model
   names, lab names, and established acronyms (MoE, MLA, GQA, RoPE, KV cache,
   DSA, MTP) in their canonical form, but everything around them in the user's
   language. Default to the language of the user's most recent message; when in
   doubt, ask once.

2. **Every parameter count comes from the official source, cited.** Total params,
   active params, expert counts, layer/head counts, and context length must be
   taken from the lab's official repo / model card / technical report
   (HuggingFace, ModelScope, or arXiv) — never from memory, never from a
   secondary blog, never estimated. For each model record the exact source URL in
   the `sources` field and reflect it in the report. If a model ships as a
   **family of variants** (e.g. Pro/Flash/Max, Air/Plus, Base/Instruct/Thinking),
   list each variant separately with its own numbers — do not collapse them or
   average them. If a number cannot be verified from an official source, write
   `未确认 / unverified` and say why, rather than inventing a figure. (Real example
   of the failure this prevents: reporting DeepSeek V4 as "800B/45B" when the
   official card lists V4-Pro = 1.6T total / 49B active and V4-Flash = 284B / 13B.)

3. **Find the newest releases before writing — the roster is a floor, not a
   ceiling.** Lab version numbers move weekly. Before any report, for each tracked
   lab run a "latest" search and check OpenRouter + HuggingFace/ModelScope for
   releases newer than your last snapshot. Recently shipped models MUST appear
   (e.g. as of mid-2026: GLM-5.2, Kimi K2.7/K2.7-Code, DeepSeek V4 Pro/Flash,
   MiniMax M3, Qwen3.6/3.7, Step 3.7 Flash, Nemotron 3 Ultra). Never ship a report
   whose newest model is older than something already public.

4. **Diagrams must be substantive and model-specific, not decorative.** The
   template's SVG components are *starting points* — every architecture deep dive
   must clone and **re-annotate** them with that model's real numbers (head
   counts, expert counts, latent dims, window sizes, sparsity) and add the
   model's specific twist. The report must also include the richer cross-cutting
   diagrams (KV-cache-vs-context curves with real numbers, a release timeline, an
   MoE sparsity scatter, and the capability comparison chart). A generic, unlabeled
   diagram is treated as missing. See `references/report-spec.md` §Diagram
   requirements and `references/diagram-cookbook.md` for build recipes.

5. **A horizontal capability comparison is mandatory.** Every full report (Mode A)
   and every comparison (Mode C) must contain a capability matrix across the
   shared benchmark dimensions (reasoning, coding, agentic/tool-use, math,
   long-context, multimodal) AND a normalized capability-vs-cost view. Pair every
   benchmark with its source and its active-params/price context, and label each
   number measured vs vendor-claimed. See `references/report-spec.md`
   §Capability comparison.

## When and how to use this skill

Use the skill in one of four modes. Detect the mode from the request, then
follow the matching workflow below.

| Mode | Trigger | Output |
|------|---------|--------|
| **A. Full landscape report** | "generate the report", "survey all models", periodic refresh | Complete HTML report (all tracked models) |
| **B. Single-model deep dive** | "deep dive on DeepSeek V4", "analyze MiMo V2" | HTML section or standalone report for one model |
| **C. Comparison** | "compare Kimi K2.6 vs GLM-5.1" | HTML comparison table + analysis |
| **D. Delta / what's new** | "what changed this month", "track new releases" | HTML changelog vs. the last snapshot |

In all modes: **research first (Step 1), analyze rigorously (Step 2), then
render (Step 3).** Never skip straight to writing prose from memory — model
facts go stale fast and the entire value of this skill is freshness and rigor.

---

## The tracked model roster

These are the priority targets. Always confirm the *current* flagship + newest
release for each lab by searching — the version numbers below are anchors, not
ground truth, and will be out of date.

| Lab / Family | Anchor flagships (verify current — these age fast) | Why tracked |
|--------------|-----------------------------------|-------------|
| **OpenAI** | GPT-5.x series (5.4/5.5), gpt-oss (open weight) | Frontier closed + open-weight baseline |
| **Anthropic / Claude** | Claude Opus 4.8 / Sonnet 4.6 / Haiku 4.5 | Frontier closed, coding/agentic leader |
| **Google / Gemini** | Gemini 3.x Pro / Flash / Flash-Lite, Gemma 4 (open) | Frontier closed + open Gemma line |
| **DeepSeek** | V4 Pro (1.6T/49B) **and** V4 Flash (284B/13B) — two variants | Most-copied open arch (MLA, MoE, DSA, CSA+HCA, mHC) |
| **GLM / Z.ai (Zhipu)** | GLM-5 → 5.1 → **5.2** (744B/~40B, MIT) | High-cadence open, Prefix-LM + DSA |
| **Kimi / Moonshot** | K2 → K2.5 → K2.6 → **K2.7 / K2.7-Code** (~1T/32B) | Trillion-param MLA MoE, agentic, now multimodal |
| **Xiaomi MiMo** | MiMo-V2-Pro / **V2.5-Pro** | Top OpenRouter usage, long-horizon agentic |
| **Qwen / Alibaba** | Qwen3 → 3.5 → **3.6/3.7 Plus**, Qwen3-Coder-Next | Most complete open family, hybrid attention |
| **StepFun** | Step 3.5 Flash → **3.7 Flash** | Efficiency-first, high tok/s, MTP-3 |
| **Mistral** | Mistral 3 Large (adopts DeepSeek arch) | Western open, DeepSeek-template adopter |
| **Meta / Llama** | Llama 4 (Scout/Maverick/Behemoth) | Western open baseline, iRoPE/MoE |
| **NVIDIA Nemotron** | Nemotron 3 Super / **Ultra** | Hybrid Mamba-Transformer, hardware-co-designed |
| **MiniMax** | M1 → M2.x → **M3** | Linear/lightning attention, 1M context, multimodal |

**Variant families**: many labs now ship a *family* per release (Pro/Flash/Max,
Air/Plus, Base/Instruct/Thinking, Coder). Treat each variant as its own row with
its own verified numbers — collapsing them is a correctness error.

Always also pull the **OpenRouter top-usage list** to catch models outside this
roster that have become important (a single high-volume app or a cheap new
preview can vault a model to #1 — note these, but flag usage-rank ≠ quality).

---

## Step 1 — Research & source collection

Collect from primary sources in roughly this priority order. Do not stop at one
source per model; cross-check architecture claims against at least the technical
report **and** the model config when both exist.

### 1.0 Discovery pass (do this BEFORE the per-model pass)

Before deep-diving known models, spend the first searches finding what's new, so
the roster reflects today, not your training data or last snapshot:
- For each tracked lab, search `"<lab> latest model <current-year>"` and check
  its HuggingFace/ModelScope org page for the newest repos.
- Pull the OpenRouter rankings + programming collection to see what's actually
  being run right now.
- Compare against your last snapshot (if any). Any model newer than your latest
  snapshot entry, or newer than the roster anchors above, MUST be added.
- Build the working list of models-to-cover (including every variant of each
  release) before writing anything.

### 1.1 Source hierarchy (best → acceptable)

1. **Official technical report / paper** — arXiv, lab GitHub, lab blog. The
   ground truth for architecture and training. Search `"<model> technical
   report"` and `"<model> arxiv"`.
2. **Model config + weights (AUTHORITATIVE for all numbers)** — HuggingFace
   `config.json` and model card; ModelScope mirror for Chinese labs (often posted
   there first/only). The config/card is the source of truth for total/active
   params, expert counts, layer/head counts, latent dims, RoPE base, and context
   length. Fetch `huggingface.co/<org>/<model>` (card) and
   `huggingface.co/<org>/<model>/blob/main/config.json`. **No parameter count
   enters the report without one of these (or the official report) behind it,
   cited.** When a release is a family, fetch each variant's card separately.
3. **Sebastian Raschka "Ahead of AI" analysis** — the best independent
   architecture explainers, with consistent diagrams. Key URLs to search for
   and fetch (they update over time):
   - `magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison`
   - `magazine.sebastianraschka.com/p/recent-developments-in-llm-architectures`
   - `magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight`
   - `magazine.sebastianraschka.com/p/technical-deepseek`
   - `sebastianraschka.com/llm-architecture-gallery/` (diagrams + fact sheets)
   - `magazine.sebastianraschka.com/p/workflow-for-understanding-llms`
4. **OpenRouter** — live usage rankings (`openrouter.ai/rankings`), the
   programming collection (`openrouter.ai/collections/programming`), per-model
   pages for **pricing incl. effective/cached pricing** and context window.
   This is the single best source for "what developers actually run" + price.
5. **Pricing pages** — official API pricing for closed models (OpenAI,
   Anthropic, Google); OpenRouter or provider pages for open models. Always
   record input / output / cached-input prices per 1M tokens.
6. **Podcasts / talks / interviews** — Raschka on Vanishing Gradients,
   lab founder interviews, conference talks (PyTorch Conf, NeurIPS). Use for
   design rationale and roadmap signals, never as the sole architecture source.
7. **Aggregators (lowest trust)** — Artificial Analysis, llm-stats, secondary
   blogs. Use only for triangulation; never cite as the primary architecture
   source.

### 1.2 Search discipline

- Use the actual current date in queries (e.g. "DeepSeek latest 2026", not a
  stale year). Lab version numbers move monthly.
- For each tracked lab, run at least: `"<lab> latest model"`, `"<flagship>
  technical report"`, and `"<flagship> config.json huggingface"`.
- When a config URL appears in results, **fetch it** — snippets lose the numbers.
- For Chinese labs (DeepSeek, GLM, Kimi, Qwen, MiMo, StepFun, MiniMax) also
  check ModelScope, as weights/cards sometimes land there first.
- Respect copyright: paraphrase technical reports, quote ≤15 words, one quote
  per source max. The numbers and architecture facts are not copyrightable; the
  prose is.

### 1.3 What to extract per model

Capture into the structured schema (see `references/data-schema.md`). At minimum:
release date, total/active params, layer count, attention type & head config,
positional-encoding strategy & context length, MoE config (experts/active/shared),
normalization, training/post-training notes, modality, license, and full pricing.

---

## Step 2 — Rigorous analysis

This is where the skill earns its keep. For background on the technical concepts
referenced below, read `references/architecture-concepts.md` — it explains
MHA/MQA/GQA/MLA, MoE routing, RoPE/YaRN/NoPE, sliding-window/sparse/lightning
attention, MTP, QK-Norm, and the KV-cache math. Read it before writing analysis
so the explanations are correct and consistent.

For each model, produce three analysis layers:

### 2.1 Capabilities (the "what")
- Benchmark posture across reasoning, coding, agentic, long-context, math,
  multilingual, multimodal — but always paired with the cost/active-params that
  produced it. A score without its price and active-param context is noise.
- Reasoning mode: dedicated reasoning, hybrid (toggle), or instruct-only.
- Agentic/tool-use posture and context window (and *effective* usable context).
- Known failure modes / caveats (hallucination rate, context degradation).

### 2.2 Architecture (the "how") — be very detailed
This is the core. For each model, explain not just *what* it uses but *why*,
and how it differs from its nearest neighbor. Cover:
- **Attention**: exact mechanism, head counts, KV-cache strategy, and the
  memory/quality tradeoff it makes. Show the KV-cache size reasoning.
- **MoE**: total vs active params, expert count/granularity, shared-expert
  choice, routing & load-balancing strategy, sparsity ratio.
- **Positional / context**: RoPE base, YaRN/NoPE/DCA, sliding-window ratio,
  sparse-attention (DSA-style) details, claimed vs usable context.
- **Efficiency tricks**: MTP, FP8/quantization, KV-sharing, compressed
  attention, per-layer embeddings, gated attention.
- **Lineage**: what it inherits from (DeepSeek V3 template? Llama? prior gen)
  and the specific deltas.

Use the analysis framing from Raschka's "workflow for understanding LLMs":
isolate *what changed inside the transformer block / residual stream / KV cache /
attention computation* and ignore the marketing.

### 2.3 Cost & economics (the "how much")
- Input / output / cached-input price per 1M tokens; note the growing dominance
  of cached-input pricing and report *effective* price where OpenRouter gives it.
- Price-per-intelligence: position the model on the cost-vs-capability frontier.
- Deployment footprint for open models (VRAM for weights at given precision;
  remember MoE total params ≠ active params for memory).

### 2.4 Cross-model synthesis
After the per-model passes, write the synthesis the report hinges on:
- The KV-cache / long-context efficiency race (who's winning, with what).
- MoE sparsification trend and the shared-expert debate.
- The attention-mechanism fork (MLA vs GQA vs linear/lightning vs sparse).
- Architecture convergence (the "DeepSeek template" spreading) vs divergence
  (linear-attention and Mamba-hybrid bets).
- Geographic / pricing shifts (e.g. Chinese-model share, price compression).

---

## Step 3 — Render the HTML report

Always read `references/report-spec.md` and `references/diagram-cookbook.md`
before generating, and base the output on `assets/report-template.html` (a
self-contained, dependency-free template with the house visual language, diagram
components, and dark/light theming).

Key rules (full detail in the spec):
- **Language follows the user** (rule #1). Localize every heading, sentence,
  table header, diagram label, and caption; keep canonical model names/acronyms.
- **Self-contained single `.html` file** — inline CSS/JS, no external build,
  no CDN dependency required for core rendering (charts may use an inlined lib).
- **Diagram-rich and model-specific** — every architecture section gets at least
  one inline SVG diagram cloned from the template and **re-annotated with that
  model's real numbers**, plus the cross-cutting diagrams (KV-cache curves,
  timeline, sparsity scatter, capability chart). Generic/unlabeled = missing.
- **Capability comparison is mandatory** — a benchmark matrix across dimensions
  plus a capability-vs-cost view (rule #5).
- **Every number sourced** — each model card links its primary sources (report,
  config/card, OpenRouter); params trace to an official source (rule #2).
- **Comparison tables** — roster-wide spec table + per-dimension comparison
  matrices (attention, MoE, context, pricing, capability).
- **Dated & versioned** — header shows generation date and a snapshot version;
  a "what changed since last snapshot" section when prior data exists.

Write the file to the outputs directory and present it. For very large reports,
build section-by-section into the template rather than one giant string.

## Step 4 — Pre-delivery QA gate

Before presenting the report, verify every item. If any fails, fix it first.

- [ ] **Language**: entire report in the user's language (rule #1); only model
      names/acronyms left canonical.
- [ ] **Newest models present**: nothing public and newer than the report's
      newest model is missing (rule #3). Re-checked OpenRouter + HF/ModelScope.
- [ ] **Variants split**: every Pro/Flash/Max/Air/Coder variant has its own row
      and numbers (rule #2).
- [ ] **Params sourced**: every total/active/expert/context number traces to an
      official card/report URL in `sources`; none from memory (rule #2). Spot-
      check the two or three highest-profile models against their HF cards.
- [ ] **Diagrams real**: each deep dive has ≥1 model-specific annotated SVG; the
      cross-cutting diagrams (KV-cache, timeline, sparsity, capability) exist and
      carry real numbers (rule #4).
- [ ] **Capability matrix**: present, multi-dimension, with sources and a
      cost-vs-capability view; measured vs claimed labeled (rule #5).
- [ ] **Renders**: open/parse the HTML; SVG tags balanced; theme toggle + sort
      work; no broken placeholders left.

---

## Recurring / periodic operation

This skill is designed to be re-run on a cadence (e.g. monthly). To support
deltas:
- Persist each run's structured data as `snapshots/<YYYY-MM-DD>.json` (schema in
  `references/data-schema.md`). If the user has a prior snapshot, load it.
- On each run, diff new data against the latest snapshot and surface: new
  releases, version bumps, architecture changes, price changes, and
  usage-rank movements.
- Keep the roster current: if a new lab/model breaks into the OpenRouter top
  list, add it to the tracked set and note it.

If the user wants true automation (scheduled runs), explain that this skill
defines the workflow and templates; the scheduling itself is handled by whatever
runs Claude (a cron-driven Claude Code job, Cowork automation, etc.). Offer to
emit a runnable script (`scripts/collect.py` is a starting point) they can
schedule.

---

## Reference files

Read these as needed; they keep this SKILL.md lean (progressive disclosure):

- `references/architecture-concepts.md` — deep, correct explanations of every
  architecture concept the analysis relies on (read before writing Step 2).
- `references/data-schema.md` — the structured JSON schema for per-model data
  and snapshots (read before Step 1 extraction and before persisting).
- `references/report-spec.md` — exact HTML report structure, sections, diagram
  requirements, capability-comparison spec, and house style (read before Step 3).
- `references/diagram-cookbook.md` — concrete recipes for the richer diagrams
  (annotated per-model block/attention/MoE views, KV-cache curves, release
  timeline, sparsity scatter, capability radar/bars). Read before Step 3.
- `references/source-directory.md` — concrete URLs and search patterns per lab
  (read during Step 1 to avoid missing sources).
- `scripts/collect.py` — optional helper to fetch OpenRouter rankings/pricing
  and HuggingFace configs into the schema; a scaffold for scheduled runs.
- `assets/report-template.html` — the self-contained report template with SVG
  diagram components and house styling.
