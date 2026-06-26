# llm-architecture-tracker

A skill that turns Claude into a disciplined analyst tracking the frontier LLM
landscape: it collects primary sources, extracts architecture + capability +
pricing details rigorously, and emits a diagram-rich HTML research report on a
recurring basis.

## What it does

- **Tracks** OpenAI, Claude, Gemini, DeepSeek, GLM, Kimi, Xiaomi MiMo, Qwen,
  StepFun, Mistral, Llama, NVIDIA Nemotron, MiniMax — plus whatever is topping
  OpenRouter usage.
- **Researches** from primary sources first: technical reports / papers,
  HuggingFace & ModelScope configs, OpenRouter usage + pricing, and Sebastian
  Raschka's architecture analyses.
- **Analyzes** at three layers — capabilities, architecture (very detailed,
  block-level, diagram-driven), and cost/economics — then synthesizes the
  cross-cutting trends.
- **Reports** as a self-contained, dependency-free HTML file with inline SVG
  architecture diagrams, sortable spec tables, dark/light theme, and a
  "what changed since last snapshot" delta.

## Four modes

| Mode | Trigger | Output |
|------|---------|--------|
| A | "generate the report" / periodic refresh | full landscape HTML |
| B | "deep dive on <model>" | single-model HTML |
| C | "compare X vs Y" | comparison HTML |
| D | "what's new this month" | delta HTML vs last snapshot |

## Structure

```
llm-architecture-tracker/
├── SKILL.md                         # entry point: workflow, roster, modes
├── references/
│   ├── architecture-concepts.md     # the technical ground truth (read before analysis)
│   ├── data-schema.md               # per-model + snapshot JSON schema
│   ├── report-spec.md               # exact HTML report structure & house style
│   └── source-directory.md          # concrete URLs + search patterns per lab
├── scripts/
│   └── collect.py                   # optional: pull OpenRouter + HF configs into a snapshot
├── assets/
│   └── report-template.html         # self-contained template + SVG diagram components
└── snapshots/                       # created at runtime: snapshots/<YYYY-MM-DD>.json
```

## Recurring use

Re-run on a cadence. Each run persists `snapshots/<date>.json`; the next run
diffs against it to surface new releases, version bumps, architecture changes,
price moves, and usage-rank shifts. The skill defines the workflow + templates;
scheduling is handled by whatever runs Claude (Claude Code cron, Cowork
automation, etc.). `scripts/collect.py` is a starting point for the automated
data pull.

## Design principles

- Primary sources over summaries; structure over vibes.
- Every benchmark paired with its cost + active-param context.
- Distinguish measured (config) vs claimed (vendor) vs inferred (analyst) facts.
- Diagrams over walls of text.
- Usage rank ≠ quality — always flagged.
