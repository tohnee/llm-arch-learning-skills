# Diagram Cookbook

Concrete recipes for every data-driven diagram and structural SVG used by the skill. Copy these shapes, helpers, and patterns directly — do not invent ad-hoc styles.

---

## 1. Structural SVGs (block diagrams of model architectures)

These SVGs appear inside model cards (small) and deep-dive pages (large). Use the **illustrative, consistent style** defined here — not photo reproductions of paper figures.

### 1.1 Shared visual grammar

```
Colors (use CSS variables from the template):
  Block fill:         var(--surface-2)
  Block stroke:       var(--line)
  Accent (attention): var(--accent)
  Accent (FFN/MoE):   var(--accent-2)  (define in template if missing, else #e9a03b / warm orange)
  Residual path:      var(--ink-soft), 1px, rounded arrows
  Text color:         var(--ink) / var(--ink-soft)
  Font:               var(--mono) for labels, var(--sans) for short captions

Shapes:
  - Norm: small (28 × 14) rounded rectangle
  - Attention: large rectangle, rounded corners (r=6), height ~80
  - FFN: same size rectangle as attention
  - MoE FFN: same outer rectangle, inside contains N small expert blocks (2–4 columns) plus a "Router" pill above
  - MTP: thin rectangle below the main block, separated by dashed line
  - Residuals: curved paths from norm input → output, labeled "+" at the add point
  - Special features (QK-Norm, SwiGLU, FP8, DSA, etc.): small pill badges
```

### 1.2 Generic Dense Pre-Norm Block (template)

```
(y increases downward; assume a 280 × 240 canvas for small, 560 × 480 for large)

[input x]
   |
   +---------------------------+
   |                           |
  [Norm]                       |
   |                           |
[Attention (GQA-8kv)]          |  (attention block in accent color)
   |   + QK-Norm pill          |
   |   + head dim 128          |
   +----------(+) <------------+
   |
   +---------------------------+
   |                           |
  [Norm]                       |
   |                           |
   [FFN (SwiGLU)]              |
   |                           |
   +----------(+) <------------+
   |
[output x]

If RoPE base is non-standard: add small badge next to Attention (e.g. "RoPE 10M")
If sliding window: add small wavy pattern on left edge of Attention
If MTP: below the block draw a dashed separator then [MTP: 1 extra head]
If Muon optimizer mentioned: add "Muon" pill near bottom
```

### 1.3 MLA (Multi-Head Latent Attention) variation

Replace the attention block contents:
- Instead of separate Q/K/V projections, show a **latent compression** step:
  - Input → "c_KV (latent dim 512)" (small rectangle)
  - then up-projection to K and V heads
  - Q is projected separately
- Label the block "MLA (latent 512)"
- Optionally show the decoupled RoPE path (Q gets RoPE; K latent does not) as a small bypass arrow

### 1.4 MoE variation

Replace the FFN block with:
- Outer FFN-sized rectangle, label "MoE FFN"
- Near top: a small rounded pill "Router"
- Below router: a grid of expert rectangles (number of columns = min(E_active, 4) or 2 for small SVGs)
  - Active experts: filled `var(--accent-2)` stroke, 100% opacity
  - Inactive experts: filled with var(--surface-2), dashed stroke, opacity 0.5
  - Shared experts (if `moe_shared_experts > 0`): drawn above the regular grid in a separate row, filled `var(--accent)`, labeled "Shared × N"
- Annotate inside or below: "N experts, E active per token, S shared" using numbers from the model entry
- Arrows from router fan out to active experts only

### 1.5 DSA (Dynamic Sparse Attention) variation

Attention block annotated "DSA (sparse)":
- Inside the attention block draw a sparse heatmap: small 8×8 grid with ~30% filled cells, rest empty
- Label: "Dynamic sparsity, block-wise"
- Mention "CSA + HCA" if DeepSeek V4 (compressive + hierarchical attention)

### 1.6 Block-diagram sizing

| Mode | Canvas | Block size | Font size |
|---|---|---|---|
| Model card (small) | 280 × 220 | attention/FFN 200 × 54 | 10–11px |
| Deep-dive (large) | 560 × 420 | attention/FFN 420 × 90 | 12–14px |
| Compare (side-by-side) | 220 × 220 each | attention/FFN 160 × 44 | 9–10px |

Always include `<title>{model.display_name} transformer block</title>` and `role="img" aria-label="...description..."` for accessibility.

---

## 2. Data-chart helpers (CH object)

All charts are produced by a tiny vanilla-JS helper object `CH` that is already defined inside `assets/report-template.html` (inline, no external deps). When generating reports from Python, your script calls these helpers and inlines the returned SVG strings into the HTML.

The four helpers below are the **only** chart types you need. If a new chart type seems required, extend the CH object inside the template and document it here — do not bring in Chart.js / D3 / etc.

### 2.1 `CH.groupedBars(data, seriesNames, opts)`

Grouped bar chart — used for capability comparison.

**Signature:**
```js
CH.groupedBars(data, series, {
  w = 720,
  h = 360,
  max = 100,      // y-axis max
  unit = '',      // suffix for y-axis labels (usually '')
  title = '',
  ylabel = 'score'
} = {})
```

**Parameters:**
- `data`: object `{ dimensionLabel: number[] }`. The number array is one score per series, in the same order as `series`.
  Example: `{ Reasoning: [92, 88, 85, 80], Coding: [90, 86, 89, 78], ... }` for six dimensions × N models.
- `series`: array of series labels (e.g. `["DeepSeek V4 Pro", "Kimi K2.7", "GLM-5.2", "Qwen3 235B"]`). One color per series.
- All dimensions MUST be exactly the canonical six (Reasoning, Coding, Agentic, Math, Long-context, Multimodal) in landscape mode. Deep-dive / compare modes use the same six.
- Returns: SVG string.

**Visual rules:**
- Groups are on the x-axis, one group per dimension.
- Within a group, bars are ordered the same as `series` across ALL groups.
- Colors are picked from a built-in palette (cycle if more than 8 series).
- Y-axis 0 → `max` with ticks every 20.
- Gridlines at ticks (light `var(--line)`).
- Legend at top.
- Values > 90 get a slightly brighter edge to highlight leaders.

---

### 2.2 `CH.scatter(points, opts)`

Scatter plot — used for architecture taxonomy (active vs total params) and cost-vs-capability.

**Signature:**
```js
CH.scatter(points, {
  w = 720,
  h = 420,
  xlabel = '',
  ylabel = '',
  title = '',
  xlog = false,     // use log scale on x (default false; set true for params and price)
  ylog = false,
  xmax = null,      // override auto-max
  ymax = null,
  parityLine = false, // draw dotted y=x line (used for param parity, Dense vs MoE)
  quadrants = false,  // draw 2×2 midlines and label quadrants (cost-vs-capability)
  colorBy = null    // optional: name of categorical field on each point to color by
} = {})
```

**`points` shape:** array of
```js
{
  x: number,
  y: number,
  label: string,            // model id or short name (drawn next to dot)
  color?: string,           // explicit color (overrides palette)
  [colorBy]?: string        // categorical value, colored by palette if colorBy is set
}
```

**Visual rules:**
- Dots are r=5 circles, filled with the series color, 1px white/ink stroke for contrast.
- Labels offset (6, -4) from the dot; if `label` is long, truncate with … and add `<title>` tooltip with the full name.
- If `xlog` / `ylog`, transform the coordinate with `Math.log10` and draw log-spaced ticks at 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000…
- If `parityLine`, draw `y = x` as a dotted line in `var(--ink-soft)` and label "Dense parity" near it.
- If `quadrants`, draw vertical + horizontal midlines (at median, or override via `opts.xMid`, `opts.yMid`) and label four quadrants in corners ("best value", "frontier premium", etc.) using small uppercase labels in `var(--ink-soft)`.
- Always leave 60px left padding and 50px bottom padding for axis labels.

---

### 2.3 `CH.lines(series, opts)`

Multi-series line chart — used for KV-cache footprint.

**Signature:**
```js
CH.lines(series, {
  w = 720,
  h = 380,
  xlabel = 'context (tokens)',
  ylabel = 'KV cache (GB)',
  title = '',
  xlog = true,
  ylog = false,
  annotate = [32768, 131072, 1048576]  // vertical annotation lines (32k, 128k, 1M)
} = {})
```

**`series` shape:** array of
```js
{
  name: string,           // model display name
  points: [ [x, y], ... ] // x = context tokens, y = GB of KV cache
  color?: string
}
```

**Data generation (in Python):** for every model in the snapshot, compute KV cache at these context lengths:
`[8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152]`
using the formula in `scripts/generate_report.py` (or this cookbook §3 below).
Clip points beyond the model's `context_window`. Use BF16 (2 bytes per element) as the default precision; note in the caption if you deviate.

**Visual rules:**
- X-axis log scale (32k → 2M).
- Y-axis starts at 0.
- Gridlines at axis ticks.
- Lines 1.5px wide; MLA/linear-attention models (nearly-flat lines) drawn 2px with a dash pattern to make their flatness obvious.
- Annotation lines at `annotate` values: thin dashed `var(--ink-soft)` verticals with small labels above ("32k", "128k", "1M").
- Legend on the **right** (reserve 90px), listing series by vertical proximity to the line end to reduce eye travel.
- A 20-GB horizontal reference line (dotted red-ish `var(--red, #d04b4b)`) labeled "20 GB / typical H100 budget"; note this is a serving heuristic, not a hard limit.

**KV-cache formula (for reference; matches `generate_report.py`):**
```
For a given sequence length L:
  if attention_type is "linear" / "lightning":
    bytes ≈ 2 * hidden_dim * num_layers * bytes_per_elt        # (O(d_model), not O(L))
  elif attention_type is "MLA" OR kv_cache_strategy is "low-rank":
    bytes ≈ 2 * num_layers * mla_latent_dim * L * bytes_per_elt
  else (MHA / GQA / MQA):
    bytes ≈ 2 * num_layers * num_kv_heads * head_dim * L * bytes_per_elt
Return GB = bytes / (1024^3)
```
The leading "2" is for K + V caches.

---

### 2.4 `CH.timeline(items, opts)`

Horizontal release timeline — used for landscape report.

**Signature:**
```js
CH.timeline(items, {
  w = 760,
  h = null,           // auto-size based on number of lanes needed
  startDate = null,   // 'YYYY-MM-DD' — auto-derived as 30 days before earliest item if null
  endDate = null      // auto-derived as 30 days after latest item if null
} = {})
```

**`items` shape:** array of
```js
{
  date: 'YYYY-MM-DD',
  label: string,      // display name
  tag: string,        // short tag: e.g. "MoE / MLA", "Dense / GQA", "linear attn"
  lane?: number       // optional; auto-assigned if omitted to avoid label overlap
}
```

**Visual rules:**
- Horizontal axis is time, left → right.
- Main spine drawn across the middle.
- Each item is a vertical tick + a flag above or below (alternate to avoid overlap; if `lane` provided, use it to group by lab).
- Flag: small rounded rectangle with `label` in bold and `tag` in smaller `var(--ink-soft)` under it.
- Month gridlines drawn in `var(--line)`.
- Models from the same family/lab share a color (assign by palette; keep consistent within a snapshot).
- Auto-assign lanes (simple greedy: place above/below spine, shift label if it would collide with a neighbor within 80px).

---

## 3. Color palette (built into CH)

```js
CH.PALETTE = [
  '#4f8cff',  // blue
  '#ff9f43',  // orange
  '#26c281',  // green
  '#ee5a6f',  // red
  '#a569bd',  // purple
  '#1abc9c',  // teal
  '#f1c40f',  // yellow
  '#95a5a6',  // grey
  '#e67e22',  // deep orange
  '#3498db',  // light blue
];
```
Cycle with `CH.PALETTE[i % CH.PALETTE.length]`.

Accessibility: this palette is designed to be distinguishable in both light and dark themes and is reasonably colorblind-friendly (avoid red-green adjacency where possible by ordering series thoughtfully).

---

## 4. Do-nots

- No external JS chart libraries, no CDNs, no `<canvas>`. Inline SVG only.
- No 3D effects, no drop shadows that obscure data.
- Never skip a model from a required chart just to make the chart pretty. If too many models (>20), rotate labels, move the legend, or increase width — but include them all.
- Do not use hard-coded hex colors in generated SVG; always reference CSS variables where possible or pull from `CH.PALETTE`.
- All axes must have labels. All charts must have a title and an `<aria-label>`.
