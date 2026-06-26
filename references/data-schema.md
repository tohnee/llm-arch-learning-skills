# LLM Architecture Tracker - Data Schema

This document defines the structured data schema for per-model extraction and snapshot persistence. All data collected during Step 1 (Research) must conform to this schema.

## Schema Version

- **Schema version**: 1.0.0
- **Last updated**: 2026-06-25

---

## Top-Level Snapshot Structure

Each snapshot is a JSON file stored at `snapshots/YYYY-MM-DD.json` with the following structure:

```json
{
  "schema_version": "1.0.0",
  "generated_at": "2026-06-25T00:00:00Z",
  "snapshot_id": "2026-06-25",
  "previous_snapshot_id": "2026-05-25",
  "models": [
    { /* ModelEntry objects */ }
  ],
  "openrouter_rankings": [
    { /* OpenRouterRanking objects */ }
  ],
  "synthesis": {
    /* CrossModelSynthesis object */
  }
}
```

---

## ModelEntry Object

Per-model structured data. All fields marked **(required)** must be populated; fields marked **(optional)** may be null if unavailable.

### Identification & Metadata

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | ✅ | Stable identifier, e.g. `"deepseek-v4"`, `"claude-opus-4"` |
| `lab` | string | ✅ | Lab/organization name, e.g. `"DeepSeek"`, `"Anthropic"` |
| `family` | string | ✅ | Model family, e.g. `"DeepSeek V"`, `"Claude"` |
| `display_name` | string | ✅ | Human-readable name, e.g. `"DeepSeek V4"` |
| `version` | string | ✅ | Exact version string, e.g. `"V4"`, `"4.5"` |
| `release_date` | string (ISO date) | ✅ | Announcement/release date, e.g. `"2026-05-20"` |
| `is_open_weight` | boolean | ✅ | true if weights are publicly downloadable |
| `license` | string | ✅ | License identifier: `"apache-2.0"`, `"mit"`, `"proprietary"`, `"llama-license"`, etc. |
| `modalities` | string[] | ✅ | Array of supported modalities: `"text"`, `"image"`, `"audio"`, `"video"`, `"code"` |

### Scale Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `total_params_b` | number | ✅ | Total parameters in billions. For MoE, this is total (not active). |
| `active_params_b` | number | ✅ | Active parameters per token in billions. Dense models: equals total_params_b. MoE: routed experts only. |
| `num_layers` | number | ✅ | Number of transformer decoder layers |
| `hidden_dim` | number | ✅ | Hidden dimension / model dimension (d_model) |
| `intermediate_dim` | number | ✅ | FFN intermediate dimension (for SwiGLU, this is usually 8/3 * hidden_dim or similar) |
| `vocab_size` | number | (optional) | Tokenizer vocabulary size |

### Attention Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `attention_type` | enum | ✅ | One of: `"MHA"`, `"MQA"`, `"GQA"`, `"MLA"`, `"linear"`, `"lightning"`, `"mamba-hybrid"` |
| `num_attention_heads` | number | ✅ | Total query heads |
| `num_kv_heads` | number | ✅ | Number of KV heads (for MQA=1, for GQA<num_heads, for MHA=num_heads) |
| `head_dim` | number | ✅ | Dimension per attention head |
| `kv_cache_strategy` | enum | ✅ | One of: `"standard"`, `"compressed"`, `"shared"`, `"low-rank"`, `"none"` |
| `qk_norm` | boolean | ✅ | Whether QK normalization is applied |
| `qk_norm_type` | string | (optional) | e.g. `"RMSNorm"`, if qk_norm is true |
| `attention_variants` | string[] | (optional) | Additional attention mechanisms: `"sliding-window"`, `"sparse-dsa"`, `"dca"`, `"linear-attn"` |

### Positional Encoding & Context

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `position_encoding` | enum | ✅ | One of: `"RoPE"`, `"ALiBi"`, `"NoPE"`, `"learned"`, `"YaRN"` |
| `rope_base` | number | (optional) | RoPE base frequency (theta), if applicable |
| `rope_scaling` | string | (optional) | RoPE scaling method: `"linear"`, `"NTK"`, `"YaRN"`, `"dynamic"`, `null` |
| `context_window` | number | ✅ | Advertised maximum context length in tokens |
| `effective_context` | number | (optional) | Verified usable context (often less than advertised) |
| `sliding_window_size` | number | (optional) | Sliding window size in tokens, if used |
| `sliding_window_ratio` | number | (optional) | Ratio of layers using sliding window (e.g. 0.5 for half layers) |

### Mixture of Experts (MoE) - null for dense models

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `is_moe` | boolean | ✅ | true if model uses MoE |
| `num_experts` | number | (optional) | Total number of routed experts (required if is_moe=true) |
| `num_active_experts` | number | (optional) | Number of experts activated per token (top-k) |
| `num_shared_experts` | number | (optional) | Number of shared experts (always active), e.g. DeepSeek shared experts |
| `expert_intermediate_dim` | number | (optional) | FFN dim per expert, if different from intermediate_dim |
| `routing_strategy` | string | (optional) | Router type: `"top-k"`, `"grouped"`, `"expert-choice"`, etc. |
| `load_balancing` | string | (optional) | Load balancing method/auxiliary loss description |
| `moe_sparsity_ratio` | number | (optional) | Active params / total params ratio |

### Normalization & Architecture Details

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `normalization_type` | enum | ✅ | One of: `"Pre-LN"`, `"Post-LN"`, `"Pre-RMSNorm"`, `"DeepNorm"`, `"SmoothQuant"` |
| `norm_epsilon` | number | (optional) | Epsilon value for normalization |
| `ffn_type` | enum | ✅ | One of: `"ReLU"`, `"GELU"`, `"SwiGLU"`, `"GeGLU"`, `"MoE-SwiGLU"` |
| `norm_position` | enum | ✅ | One of: `"pre-norm"`, `"post-norm"`, `"pre-norm-with-embed-norm"` |
| `residual_scaling` | number | (optional) | Residual scaling factor (e.g. for DeepNorm) |

### Training & Post-Training

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `training_tokens` | number | (optional) | Total pre-training tokens in trillions |
| `training_data_cutoff` | string (ISO date) | (optional) | Training data cutoff date |
| `post_training_methods` | string[] | (optional) | Methods used: `"SFT"`, `"RLHF"`, `"DPO"`, `"GRPO"`, `"RLVR"`, `"RLOO"` |
| `reasoning_mode` | enum | ✅ | One of: `"none"`, `"dedicated-reasoning"`, `"hybrid"`, `"instruct-only"` |
| `reasoning_budget_tokens` | number | (optional) | Default/available thinking token budget |
| `supports_mtp` | boolean | ✅ | Whether Multi-Token Prediction is supported |
| `mtp_future_tokens` | number | (optional) | Number of future tokens predicted in MTP |
| `quantization_support` | string[] | (optional) | Supported quantization: `"FP8"`, `"INT8"`, `"INT4"`, `"AWQ"`, `"GPTQ"` |

### Performance & Capabilities

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `benchmark_scores` | object | (optional) | Key-value pairs of benchmark name → score |
| `strengths` | string[] | ✅ | Key capability strengths (e.g. `["coding", "agentic", "math"]`) |
| `weaknesses` | string[] | ✅ | Known limitations/failure modes |
| `throughput_tok_s` | number | (optional) | Measured throughput in tokens/second on reference hardware |
| `ttft_ms` | number | (optional) | Time to first token (median) |
| `tpot_ms` | number | (optional) | Time per output token (median) |

### Pricing & Economics

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `pricing_input_per_m` | number | ✅ | Input price per 1M tokens in USD |
| `pricing_output_per_m` | number | ✅ | Output price per 1M tokens in USD |
| `pricing_cached_input_per_m` | number | (optional) | Cached input price per 1M tokens (prompt caching) |
| `pricing_effective_per_m` | number | (optional) | Effective price per 1M tokens (OpenRouter blended) |
| `pricing_notes` | string | (optional) | Pricing notes (tiers, free quota, etc.) |
| `price_per_intelligence` | enum | (optional) | One of: `"best-value"`, `"premium"`, `"budget"`, `"frontier"` |

### Deployment Footprint (open models only)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `vram_fp8_gb` | number | (optional) | VRAM required for FP8 inference in GB |
| `vram_fp16_gb` | number | (optional) | VRAM required for FP16/BF16 inference in GB |
| `vram_int4_gb` | number | (optional) | VRAM required for INT4 quantized inference in GB |
| `recommended_hardware` | string | (optional) | Recommended GPU(s) for deployment |

### Sources

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sources` | object | ✅ | See SourceLinks below |
| `notes` | string[] | (optional) | Free-text analyst notes |

---

## SourceLinks Object

Primary source links for verification.

```json
{
  "technical_report": "https://arxiv.org/abs/xxxx.xxxxx",
  "huggingface_config": "https://huggingface.co/org/model/blob/main/config.json",
  "huggingface_card": "https://huggingface.co/org/model",
  "modelscope_card": "https://modelscope.cn/models/org/model",
  "openrouter_page": "https://openrouter.ai/model/id",
  "official_pricing": "https://api.example.com/pricing",
  "lab_blog": "https://lab.example.com/blog/model-announcement",
  "additional_sources": [
    "https://...",
    "https://..."
  ],
  "last_verified": "2026-06-25"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `technical_report` | string | (optional) | arXiv or official technical report URL |
| `huggingface_config` | string | (optional) | Direct link to config.json |
| `huggingface_card` | string | (optional) | HuggingFace model card URL |
| `modelscope_card` | string | (optional) | ModelScope mirror URL (Chinese labs) |
| `openrouter_page` | string | (optional) | OpenRouter model page for pricing/usage |
| `official_pricing` | string | (optional) | Official API pricing page |
| `lab_blog` | string | (optional) | Official lab blog announcement |
| `additional_sources` | string[] | (optional) | Other corroborating sources |
| `last_verified` | string (ISO date) | ✅ | Date when this data was last cross-checked |

---

## OpenRouterRanking Object

Usage data from OpenRouter rankings page.

```json
{
  "model_id": "deepseek/deepseek-chat",
  "display_name": "DeepSeek V3",
  "usage_rank": 3,
  "when_to_use_score": 9.2,
  "throughput_tok_s": 85,
  "pricing_input_per_m": 0.27,
  "pricing_output_per_m": 1.10,
  "context_window": 128000,
  "endorsed_by": "deepseek"
}
```

---

## CrossModelSynthesis Object

Cross-cutting analysis generated after per-model extraction.

```json
{
  "architecture_trends": [
    { "trend": "string", "evidence_models": ["model-id-1", "model-id-2"], "description": "string" }
  ],
  "attention_landscape": {
    "mha_models": [],
    "gqa_models": [],
    "mla_models": [],
    "linear_attn_models": [],
    "mamba_hybrid_models": []
  },
  "moe_trends": {
    "num_moe_models": 0,
    "avg_experts": 0,
    "shared_experts_adoption": 0.5,
    "typical_sparsity_ratio": 0.1
  },
  "context_race": {
    "longest_advertised": { "model_id": "", "context": 0 },
    "longest_verified": { "model_id": "", "context": 0 },
    "rope_yarn_users": []
  },
  "pricing_trends": {
    "cheapest_per_m_input": { "model_id": "", "price": 0 },
    "best_value_frontier": [{"model_id": "", "quality_score": 0, "price_per_m": 0}],
    "price_compression_vs_last": 0.0
  },
  "notable_changes_vs_previous": [
    { "change_type": "new-release|price-drop|arch-change|version-bump", "model_id": "", "description": "" }
  ]
}
```

---

## Validation Rules

Required fields must not be null or empty. When a field is marked (optional) but data is available, it **must** be populated.

### Cross-Field Invariants

1. If `is_moe` = true, then `num_experts`, `num_active_experts` must be present, and `active_params_b` < `total_params_b`.
2. If `is_moe` = false (dense), then all MoE fields must be null.
3. For attention types:
   - `MHA`: `num_kv_heads` == `num_attention_heads`
   - `MQA`: `num_kv_heads` == 1
   - `GQA`: 1 < `num_kv_heads` < `num_attention_heads`
4. If `position_encoding` = `"RoPE"`, `rope_base` must be present.
5. If `sliding_window_size` is set, `attention_variants` must include `"sliding-window"`.
6. `release_date` must not be in the future.
7. All prices must be non-negative.
8. `effective_context` ≤ `context_window` (when both present).

### Example Minimal Valid Entry

```json
{
  "id": "example-model",
  "lab": "Example Lab",
  "family": "Example",
  "display_name": "Example Model V1",
  "version": "V1",
  "release_date": "2026-06-01",
  "is_open_weight": true,
  "license": "apache-2.0",
  "modalities": ["text", "code"],
  "total_params_b": 7,
  "active_params_b": 7,
  "num_layers": 32,
  "hidden_dim": 4096,
  "intermediate_dim": 14336,
  "attention_type": "GQA",
  "num_attention_heads": 32,
  "num_kv_heads": 8,
  "head_dim": 128,
  "kv_cache_strategy": "standard",
  "qk_norm": true,
  "position_encoding": "RoPE",
  "rope_base": 10000,
  "context_window": 128000,
  "is_moe": false,
  "normalization_type": "Pre-RMSNorm",
  "ffn_type": "SwiGLU",
  "norm_position": "pre-norm",
  "reasoning_mode": "instruct-only",
  "supports_mtp": false,
  "strengths": ["coding"],
  "weaknesses": [],
  "pricing_input_per_m": 0.15,
  "pricing_output_per_m": 0.60,
  "sources": {
    "last_verified": "2026-06-25"
  }
}
```
