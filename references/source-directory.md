# Source Directory

Concrete sources and search patterns for each tracked lab. Use during Step 1.
Version numbers age fast — always run the "latest" search first to find the
*current* flagship, then go to its report and config. Fetch config URLs; don't
trust snippets for numbers.

## Cross-lab / landscape sources

- **OpenRouter rankings** — `openrouter.ai/rankings` (overall usage),
  `openrouter.ai/collections/programming` (coding usage). Per-model pages give
  context window + input/output/**cached/effective** pricing per provider.
- **Sebastian Raschka "Ahead of AI"** — the best independent architecture
  analysis (consistent diagrams, block-level rigor). Search and fetch:
  - the Big LLM Architecture Comparison (the master reference, updated over time)
  - Recent Developments in LLM Architectures (KV sharing, mHC, compressed attn)
  - A Dream of Spring for Open-Weight LLMs (Jan–Feb 2026 roundup)
  - A Technical Tour of the DeepSeek Models from V3 to V3.2
  - the LLM Architecture Gallery at `sebastianraschka.com/llm-architecture-gallery/`
    (fact sheets + clickable diagrams + side-by-side diff tool)
  - My Workflow for Understanding LLM Architectures (the analysis method)
  - Beyond Standard LLMs (linear-attention hybrids, diffusion LLMs, world models)
- **Aggregators (triangulate only)** — Artificial Analysis, llm-stats.com,
  Emergent Mind topic pages. Never the sole architecture source.

## Per-lab

### OpenAI (GPT, gpt-oss)
- Search: `OpenAI latest model`, `GPT-5 system card`, `gpt-oss architecture`.
- Sources: openai.com/research + system cards; gpt-oss on HuggingFace
  (`huggingface.co/openai`). Pricing: `openai.com/api/pricing` + OpenRouter.
- Note: closed flagships expose little architecture; rely on system card +
  Raschka's reverse-engineering for gpt-oss.

### Anthropic (Claude)
- Search: `Claude latest model`, `Claude model card`, `Anthropic pricing`.
- Sources: anthropic.com/news + model cards; `docs.claude.com`. Pricing:
  `anthropic.com/pricing` + OpenRouter. Architecture is undisclosed — analyze
  capabilities, pricing, agentic/coding posture, context behavior.

### Google (Gemini, Gemma)
- Search: `Gemini latest`, `Gemma 4 technical report`, `Gemma config huggingface`.
- Sources: deepmind.google/models; Gemma on HF (`huggingface.co/google`) with
  configs; Gemma tech reports on arXiv. Gemma is the open, analyzable line
  (KV-sharing, PLE, sliding-window ratios). Pricing: ai.google.dev/pricing + OR.

### DeepSeek
- Search: `DeepSeek V4 technical report`, `DeepSeek latest`, `DeepSeek config`.
- Sources: `github.com/deepseek-ai`; arXiv (V3: 2412.19437; V3.2 report in repo);
  HF `huggingface.co/deepseek-ai/<model>/blob/main/config.json` and `model.py`
  for sparse-attention code. ModelScope mirror. The architecture template most
  others copy — study deltas carefully.

### GLM / Z.ai (Zhipu)
- Search: `GLM-5 technical report`, `Z.ai GLM latest`, `GLM config`.
- Sources: `github.com/zai-org/GLM-4.5` (ARC report), `github.com/zai-org/GLM-V`
  (multimodal); HF `huggingface.co/zai-org`; build.nvidia.com NIM cards have spec
  sheets (params, context). High release cadence — re-check monthly.

### Kimi / Moonshot
- Search: `Kimi K2 technical report`, `Moonshot Kimi latest`, `Kimi config`.
- Sources: arXiv (K2: 2507.20534; K2.5: 2602.02276); HF
  `huggingface.co/moonshotai`. Scaled DeepSeek template + Muon/QK-Clip — the
  report is unusually detailed on scaling-law rationale.

### Xiaomi MiMo
- Search: `Xiaomi MiMo V2 technical report`, `MiMo-V2-Pro`, `MiMo config`.
- Sources: `github.com/XiaomiMiMo`; HF `huggingface.co/XiaomiMiMo`; ModelScope.
- Note: a top OpenRouter-usage model — strong agentic/long-horizon claims;
  verify active params and attention type from the config, not the press.

### Qwen / Alibaba
- Search: `Qwen3 technical report`, `Qwen latest`, `Qwen3-Next config`,
  `Qwen3-Coder-Next tech report`.
- Sources: arXiv (Qwen3: 2505.09388; Qwen3-VL: 2511.21631);
  `github.com/QwenLM`; HF `huggingface.co/Qwen`; ModelScope. The most complete
  open family (dense + MoE + coder + VL + omni). Watch the hybrid-attention
  "-Next" line and the think/no-think framework.

### StepFun
- Search: `StepFun Step 3.5 Flash technical report`, `Step latest`, `Step config`.
- Sources: arXiv (Step 3.5: 2602.10604); HF `huggingface.co/stepfun-ai`. Efficiency
  story: small active params, high tok/s, MTP-3, gated attention.

### Mistral
- Search: `Mistral 3 Large`, `Mistral latest`, `Mistral config`.
- Sources: mistral.ai/news; HF `huggingface.co/mistralai`. Mistral 3 adopts the
  DeepSeek V3 architecture (coarsened experts) — note the Western adoption.

### Meta / Llama
- Search: `Llama 4 model card`, `Llama latest`, `Llama config`.
- Sources: ai.meta.com/blog + llama.com; HF `huggingface.co/meta-llama`. Llama 4
  Scout/Maverick/Behemoth — MoE + iRoPE + early-fusion multimodal.

### NVIDIA Nemotron
- Search: `Nemotron 3 Ultra`, `Nemotron 3 Super technical report`,
  `Nemotron config`.
- Sources: build.nvidia.com NIM cards; `huggingface.co/nvidia`; NVIDIA tech blog.
  Hybrid Mamba-2 + Transformer, hardware-co-designed, long context, aggressive
  pricing tier.

### MiniMax
- Search: `MiniMax M3 technical report`, `MiniMax latest`, `MiniMax config`.
- Sources: arXiv (M1: 2506.13585; MiniMax-01 report on minimax.io); HF
  `huggingface.co/MiniMaxAI`; minimax.io/news. The lightning/linear-attention
  hybrid leader; 1M context; CISPO RL.

## Search hygiene

- Always include the real current year/month; never a stale year.
- Per lab, minimum three queries: `<lab> latest model`, `<flagship> technical
  report`, `<flagship> config.json huggingface`.
- Fetch the config and the report; cross-check the params/heads/experts between
  them. If they disagree, prefer config for structure, report for rationale, and
  note the discrepancy.
- Chinese labs: also try ModelScope (`modelscope.cn/models/<org>/<model>`).
- Copyright: paraphrase; quote ≤15 words; one quote per source. Facts/numbers are
  free to report.
