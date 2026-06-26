#!/usr/bin/env python3
"""
collect.py — data-collection helper for the llm-architecture-tracker skill.

Pulls machine-readable data (OpenRouter model list + pricing, HuggingFace configs)
into the snapshot schema defined in references/data-schema.md.

This is the data-collection half of the pipeline; the architecture analysis
half is done by the LLM using the skill's reference files.

Network notes:
- OpenRouter public models endpoint: https://openrouter.ai/api/v1/models
  (no key required; set OPENROUTER_API_KEY env var for authenticated access).
- HuggingFace config: https://huggingface.co/<org>/<model>/raw/main/config.json
- ModelScope config: https://modelscope.cn/models/<org>/<model>/resolve/master/config.json

Usage:
    python collect.py --out snapshots/$(date +%F).json
    python collect.py --out snapshots/2026-06-25.json --hf deepseek-ai/DeepSeek-V3.2 Qwen/Qwen3-235B-A22B
    OPENROUTER_API_KEY=sk-... python collect.py --out snapshots/$(date +%F).json

Dependencies: only the standard library (urllib).
"""

import argparse
import datetime as dt
import json
import os
import sys
import urllib.request
import urllib.error

SCHEMA_VERSION = "1.0.0"
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"
HF_CONFIG = "https://huggingface.co/{repo}/raw/main/config.json"
MODELSCOPE_CONFIG = "https://modelscope.cn/models/{repo}/resolve/master/config.json"

# Labs we care about, used to tag/filter the OpenRouter list.
TRACKED_LABS = [
    "openai", "anthropic", "google", "deepseek", "z-ai", "glm", "zhipu",
    "moonshot", "kimi", "xiaomi", "mimo", "qwen", "alibaba", "stepfun", "step",
    "mistral", "meta", "llama", "nvidia", "nemotron", "minimax",
]


def fetch(url, timeout=30, api_key=None):
    """Fetch a URL with optional bearer-token auth. Returns text or None."""
    headers = {"User-Agent": "llm-arch-tracker/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        print(f"[warn] HTTP {e.code} for {url}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] failed {url}: {e}", file=sys.stderr)
    return None


def get_openrouter_models(api_key=None):
    """Return a list of normalized OpenRouterRanking objects (per data-schema.md)."""
    raw = fetch(OPENROUTER_MODELS, api_key=api_key)
    if not raw:
        return []
    data = json.loads(raw).get("data", [])
    out = []
    for idx, m in enumerate(data):
        pricing = m.get("pricing", {}) or {}

        # OpenRouter prices are per-token strings; convert to per-1M tokens.
        def per_m(key):
            v = pricing.get(key)
            try:
                return round(float(v) * 1_000_000, 4) if v not in (None, "") else None
            except (TypeError, ValueError):
                return None

        entry = {
            "model_id": m.get("id"),
            "display_name": m.get("name"),
            "usage_rank": idx + 1,  # OpenRouter returns in popularity order
            "context_window": m.get("context_length"),
            "pricing_input_per_m": per_m("prompt"),
            "pricing_output_per_m": per_m("completion"),
            "pricing_cached_input_per_m": per_m("input_cache_read"),
            "modality": (m.get("architecture", {}) or {}).get("modality"),
            "endorsed_by": m.get("organization"),
            "source_url": f"https://openrouter.ai/{m.get('id', '')}",
        }
        out.append(entry)
    return out


def tag_tracked(rankings):
    """Mark models that belong to a tracked lab for prioritization."""
    for r in rankings:
        mid = (r.get("model_id") or "").lower()
        r["tracked"] = any(lab in mid for lab in TRACKED_LABS)
    return rankings


def get_hf_config(repo):
    """Fetch a HuggingFace config.json, falling back to ModelScope."""
    raw = fetch(HF_CONFIG.format(repo=repo))
    if not raw:
        raw = fetch(MODELSCOPE_CONFIG.format(repo=repo))
    if not raw:
        return None
    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError:
        return None
    # Architecture-relevant fields from data-schema.md ModelEntry.
    keep = {
        "model_type", "architectures", "num_hidden_layers", "hidden_size",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "vocab_size", "max_position_embeddings", "rope_theta",
        "intermediate_size", "n_routed_experts", "num_experts",
        "num_experts_per_tok", "n_shared_experts", "moe_intermediate_size",
        "kv_lora_rank", "q_lora_rank", "qk_rope_head_dim", "v_head_dim",
        "sliding_window", "attention_dropout", "torch_dtype",
    }
    extracted = {k: cfg.get(k) for k in keep if k in cfg}
    return {
        "repo": repo,
        "config_url": HF_CONFIG.format(repo=repo),
        "config_excerpt": extracted,
    }


def main():
    ap = argparse.ArgumentParser(description="Collect LLM landscape data into a snapshot.")
    ap.add_argument("--out", default=f"snapshots/{dt.date.today().isoformat()}.json",
                    help="Output snapshot JSON path")
    ap.add_argument("--hf", nargs="*", default=[],
                    help="HuggingFace repos to pull configs for, e.g. deepseek-ai/DeepSeek-V3.2")
    ap.add_argument("--no-openrouter", action="store_true",
                    help="Skip OpenRouter data collection")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")

    today = dt.date.today().isoformat()
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "snapshot_id": today,
        "previous_snapshot_id": None,  # filled in by detect_new.py when diffing
        "generated_by": "llm-architecture-tracker/collect.py",
        "openrouter_rankings": [],
        "hf_configs": [],
        "models": [],   # filled by the analyst/Claude with full ModelEntry objects
        "synthesis": {},
        "notes": (
            "Machine-collected scaffold. Architecture/capability analysis is "
            "done by Claude using the skill's reference files; this file only "
            "seeds pricing/context/config numbers."
        ),
    }

    if not args.no_openrouter:
        auth_note = " (authenticated)" if api_key else " (public)"
        print(f"[info] fetching OpenRouter model list{auth_note}…", file=sys.stderr)
        snapshot["openrouter_rankings"] = tag_tracked(get_openrouter_models(api_key=api_key))
        print(f"[info] got {len(snapshot['openrouter_rankings'])} models", file=sys.stderr)
        tracked_count = sum(1 for r in snapshot["openrouter_rankings"] if r.get("tracked"))
        print(f"[info] {tracked_count} models from tracked labs", file=sys.stderr)

    for repo in args.hf:
        print(f"[info] fetching config for {repo}…", file=sys.stderr)
        cfg = get_hf_config(repo)
        if cfg:
            snapshot["hf_configs"].append(cfg)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
