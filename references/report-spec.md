# Report Specification

This file defines exactly what every output HTML must contain.

## 1. Language Rule (IMPORTANT — rule #1)
The **entire report narrative, section titles, captions, and prose comments** must be written in **the same language as the user's request**. If the user asks in Chinese → all text in Chinese; English → all English; etc.

**EXCEPTION — the following remain in their canonical form and are NEVER translated:**
- Model identifiers (`model.id` strings) — e.g. `deepseek-v3-0324`, `llama-3.1-405b`
- Canonical model display names (`model.display_name`) that are established names, e.g. "DeepSeek V3.2", "Llama 3.1 405B", "Qwen3 235B-A22B", "GLM-4.5", "Mistral Large 2". These are proper nouns / brand names, translate nothing in them.
- Acronyms: MoE, GQA, MQA, MLA, MTP, FP8, BF16, SwiGLU, RoPE, RMSNorm, Prefix-LM, DSA, Muon, NTP, DCA, CISR, CSA, HCA, mHC
- Lab names: DeepSeek, Meta, Alibaba/Qwen, Zhipu/GLM, Moonshot/Kimi, Anthropic, OpenAI, xAI, Google/Gemini, Mistral
- Benchmark names: AIME, MMLU, MATH-500, HumanEval, LiveCodeBench, SWE-bench, MMMU, etc.
- Licenses: Apache 2.0, MIT, Llama 3 Community, Gemma Terms, Proprietary
- Section headings when they are fixed structural labels may be translated. "Executive Summary" ↔ "执行摘要"; "KV-Cache Footprint" ↔ "KV 缓存占用"; "Architecture Taxonomy" ↔ "架构分类"; "Model Cards" ↔ "模型卡片"; "Timeline" ↔ "时间线"; "Cross-cutting Analysis" ↔ "横向对比分析"; "Methodology" ↔ "方法论". All model-name occurrences within headings stay canonical.

In short: **free text in user language; proper nouns, model names, and technical jargon unchanged.**

---

## 2. Modes

### 2a. Full Landscape Report (`report` mode — default)
Input: a snapshot JSON (`$snapshot_path`, e.g. `snapshots/2026-03-03.json`).
Render one HTML page covering **every model in that snapshot**.

Mandatory sections, in order:

1. **Title block** — title, snapshot date (formatted long-form; in Chinese context e.g. "2026 年 3 月 3 日快照"), generation timestamp, model count.
2. **Executive summary** (narrative, user language) — 3–8 short paragraphs covering:
   - which new models / versions landed since the previous snapshot (if `meta.previous_snapshot_id` exists)
   - the biggest architectural shifts this cycle (e.g. "linear attention spreading beyond Kimi", "FP8 training becoming default", "shared-expert MoE is now standard")
   - licensing trend (more open / more gated?)
   - a 1–2 sentence "who's leading on what" synthesis (reasoning, coding, long context, cost)
3. **Capability comparison** — one grouped bar chart showing all models side-by-side across six capability dimensions (reasoning, coding, agentic, math, long-context, multimodal) on a 0–100 scale. Bars are **grouped by dimension**, one color per model, sorted within each group by score descending or by the canonical ordering (see below). The chart helper is `CH.groupedBars(...)` defined in the template (see `diagram-cookbook.md` §2.1). A short paragraph in user language summarising who leads where.
4. **KV-cache footprint** — one multi-series line chart (`CH.lines`) plotting KV cache size (GB, BF16) vs. context length (32k → 2M) for every model. **No model may be omitted**. One line per model, distinct colors, labelled legend on the right. 32k / 128k / 1M annotation lines. Short commentary noting which models are >20GB at 128k (attention needed for serving) and which are nearly flat (linear / MLA).
5. **Architecture taxonomy** — one scatter plot (`CH.scatter`) with x = **active parameters (B)** (log scale) and y = **total parameters (B)** (log scale). Each dot one model, labeled, color-coded by attention type. Dotted parity line separates Dense (on the line) from MoE (above the line). Short paragraph explaining the three architectural clusters (dense small, dense frontier, MoE frontier).
6. **Release timeline** — one horizontal timeline (`CH.timeline`) sorted left-to-right by `release_date`. Each flag shows the model display name + one tag (e.g. "MoE / MLA / 1M ctx"). Group multiple releases from the same lab vertically to avoid overlap.
7. **Cost vs. capability quadrant** — one scatter plot (`CH.scatter`) with x = **output price per 1M tokens ($)** (log scale) and y = **composite capability score** (average of the six dimensions). Each dot labeled. Four quadrants: cheap+weak, expensive+weak, cheap+strong ("best value" — highlight), expensive+strong ("frontier premium"). Short commentary on value leaders.
8. **Model cards** — for each model:
   - h2 with `display_name` + lab badge
   - one small inline SVG of the model's block structure, using the "illustrative" style:
     - generic Transformer block outline (pre-norm, attention + FFN residuals)
     - attention block labeled with the actual type (`GQA-8kv`, `MLA (latent 512)`, `MQA`, `MHA`, etc.)
     - if `is_moe=true`, replace the single FFN block with an MoE block showing N experts + E active + any shared experts (all numbers drawn from the model entry)
     - if KV strategy is non-standard (sliding window, low-rank), add a small visual hint
     - mark special features (MTP block, QK-Norm, SwiGLU/GEGLU, Muon optimizer note, Prefix-LM causal mask, DSA, FP8 training)
     Do **not** draw pixel-perfect reproductions of the paper figures — use the consistent illustrative style defined in `diagram-cookbook.md §1`.
   - a quick-facts table (release date, license, params active/total, context, architecture tokens)
   - capability mini-bars (six bars, same scale as the global comparison)
   - pricing line: input $ / output $ per 1M tokens
   - 2–5 sentence prose analysis in the user's language — what this model is known for, what architectural choices are notable, where it sits in the landscape
   - source links
9. **Cross-cutting analysis** (narrative) — 3–5 themed subsections drawing on `architecture-concepts.md`:
   - Attention: who is using what (MHA / GQA / MQA / MLA / linear / DSA / hybrid) and why, adoption trends
   - MoE: shared-expert convergence, typical expert counts, routing debates, DeepSeek V4's CSA+HCA
   - Long context: strategies in use (sliding window attention, YaRN/NoPE/RoPE scaling, sparse attention), what actually works at 1M+
   - Reasoning / MTP: who does explicit reasoning (OOD, chain-of-thought training), multi-token prediction adoption, performance impact
   - Normalization / position tricks: RMSNorm vs DeepNorm, QK-Norm, RoPE bases, Muon optimizer spread
   - Open-weights vs closed: boundary, licensing trends, Apache 2.0 / MIT returns
10. **Methodology** — short paragraph describing data sources, how capability scores are derived (normalize benchmarks to 0–100 per category, weighted average; cite that scores are estimates as of snapshot date), and a disclaimer.

### 2b. Single-model Deep Dive (`deep-dive` mode)
Input: `$snapshot_path` + `$model_id`.

Sections:
1. Title block with display name + snapshot date
2. One large, detailed SVG block diagram of the model's architecture (attention, FFN/MoE, extras like MTP), drawn in the same illustrative style but bigger and fully labelled (see §1 for style rules)
3. Parameter table (every field from the JSON that is not null)
4. Full-width capability bars (six dimensions, 0–100)
5. KV-cache line chart for this single model (same axes as landscape report, 32k → max context of this model)
6. Architecture narrative (3–8 paragraphs): what's novel, what it borrows, what it pioneered, trade-offs the architects made
7. Performance summary: headline benchmarks, pricing, value positioning
8. Source links

### 2c. Comparison (`compare` mode)
Input: `$snapshot_path` + list of `$model_ids` (2–5).

Sections:
1. Title block listing compared models
2. Side-by-side architecture SVGs (one per model, same vertical scale so attention/FFN blocks line up)
3. Head-to-head comparison table (params, context, attention type, MoE config, license, pricing)
4. Grouped capability bar chart grouped by dimension (same six dimensions, one bar group per model)
5. KV-cache line chart with one line per model (same axes)
6. Cost-vs-capability scatter restricted to the compared models, plus a note where they sit relative to the field
7. Narrative comparison (2–5 paragraphs in user language): architectural similarities and differences, trade-offs, who should use which

### 2d. Delta / What's New (`delta` mode)
Input: `$snapshot_path` + `$previous_snapshot_path`.

Sections:
1. Title block with both dates
2. List of **new models** (present in new, not in old) — for each: display name, 1-line arch summary, 1-line why it matters
3. List of **updated models** (present in both but any of: version changed, params/pricing changed, architecture fields changed, benchmark scores shifted > some threshold) — summarize what changed per model
4. List of **removed / deprecated** models, with reason if known
5. Architectural shift narrative (2–4 paragraphs): what's different in how models are being built compared to last snapshot
6. Price-change table if any model's pricing moved >10%
7. Ranking shifts: which models moved up/down in the composite capability ranking by >2 positions

## 3. Styling rules (all modes)

- Use the CSS variables already defined in `assets/report-template.html` — no hard-coded colors.
- Respect `prefers-color-scheme` (already handled by the template).
- All SVG charts must be inline SVG (no external images, no `<canvas>`, no JS libraries loaded from CDNs).
- All numerical outputs must be formatted consistently (see `data-schema.md §4`).
- Every diagram must have an `<title>` and an `aria-label` for accessibility.
- No emoji in formal labels (may be used in narrative sparingly).

## 4. Required charts at a glance

| Chart | Helper | Landscape | Deep-dive | Compare |
|---|---|---|---|---|
| Capability grouped bars | `CH.groupedBars` | ✅ all models | ✅ single | ✅ selected |
| KV-cache lines | `CH.lines` | ✅ all models | ✅ single | ✅ selected |
| Architecture scatter (active vs total params) | `CH.scatter` | ✅ all models | ❌ | ❌ |
| Timeline | `CH.timeline` | ✅ all models | ❌ | ❌ |
| Cost vs capability | `CH.scatter` | ✅ all models | ❌ | ✅ selected |
| Block-structure SVG (per model) | hand-built per rules in diagram-cookbook §1 | ✅ small per model | ✅ one large | ✅ side-by-side |

See `diagram-cookbook.md` for exact signatures and examples of each helper.
