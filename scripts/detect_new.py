#!/usr/bin/env python3
"""
detect_new.py — detect new models and significant changes between snapshots.

Compares the two most recent snapshots in a directory and produces a
machine-readable change summary: new models, version bumps, price changes,
and rank shifts. The output drives the automated workflow: if has_changes
is true, the LLM analysis step is triggered.

Usage:
    python detect_new.py --snapshot-dir snapshots/ --out snapshots/2026-06-25_changes.json
    python detect_new.py --old snapshots/2026-06-24.json --new snapshots/2026-06-25.json --out changes.json
"""

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys

PRICE_CHANGE_THRESHOLD = 0.10  # 10%
RANK_CHANGE_THRESHOLD = 5      # 5 positions


def load_snapshot(path):
    """Load a snapshot JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_snapshots(snapshot_dir, count=2):
    """Find the most recent snapshot files by filename date."""
    pattern = os.path.join(snapshot_dir, "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].json")
    files = sorted(glob.glob(pattern))
    return files[-count:] if len(files) >= count else files


def extract_version(display_name):
    """
    Try to extract a version string from a model display name.
    Returns version string or None.
    Examples:
      "DeepSeek V3" -> "V3"
      "Qwen3 72B" -> "3"
      "Claude 3.5 Sonnet" -> "3.5"
    """
    if not display_name:
        return None
    # Look for common version patterns
    patterns = [
        r"[Vv](\d+(?:\.\d+)*)",          # V3, v3.2, V4.5
        r"(\d+\.\d+(?:\.\d+)*)\s*[A-Z]",  # 3.5 Sonnet, 4.7 Pro
        r"\b(\d+)\s*[Bb]\b",              # 72B, 235B (param size, not version)
    ]
    for pat in patterns[:2]:  # skip param-size pattern for version
        m = re.search(pat, display_name)
        if m:
            return m.group(1) if m.group(1).isdigit() or "." in m.group(1) else None
    return None


def get_rank_map(rankings):
    """Build {model_id: usage_rank} dict from openrouter_rankings list."""
    result = {}
    for r in rankings:
        mid = r.get("model_id")
        rank = r.get("usage_rank")
        if mid and rank:
            result[mid] = rank
    return result


def get_price_map(rankings):
    """Build {model_id: {input, output, cached}} dict from rankings."""
    result = {}
    for r in rankings:
        mid = r.get("model_id")
        if mid:
            result[mid] = {
                "input": r.get("pricing_input_per_m"),
                "output": r.get("pricing_output_per_m"),
                "cached": r.get("pricing_cached_input_per_m"),
            }
    return result


def get_name_map(rankings):
    """Build {model_id: display_name} dict."""
    return {r["model_id"]: r.get("display_name", "") for r in rankings if r.get("model_id")}


def detect_changes(old_snap, new_snap, price_threshold=PRICE_CHANGE_THRESHOLD, rank_threshold=RANK_CHANGE_THRESHOLD):
    """
    Compare two snapshots and return a change summary.

    Returns a dict with:
      check_date, has_changes, new_models, version_bumps,
      price_changes, rank_changes
    """
    old_rankings = old_snap.get("openrouter_rankings", [])
    new_rankings = new_snap.get("openrouter_rankings", [])

    old_ids = set(r.get("model_id") for r in old_rankings if r.get("model_id"))
    new_ids = set(r.get("model_id") for r in new_rankings if r.get("model_id"))

    old_prices = get_price_map(old_rankings)
    new_prices = get_price_map(new_rankings)
    old_ranks = get_rank_map(old_rankings)
    new_ranks = get_rank_map(new_rankings)
    old_names = get_name_map(old_rankings)
    new_names = get_name_map(new_rankings)

    # --- New models ---
    new_model_ids = sorted(new_ids - old_ids)
    new_models = []
    for mid in new_model_ids:
        r = next((x for x in new_rankings if x.get("model_id") == mid), {})
        new_models.append({
            "model_id": mid,
            "display_name": r.get("display_name", ""),
            "pricing_input_per_m": r.get("pricing_input_per_m"),
            "pricing_output_per_m": r.get("pricing_output_per_m"),
            "context_window": r.get("context_window"),
        })

    # --- Version bumps (only for models that exist in both) ---
    common_ids = old_ids & new_ids
    version_bumps = []
    for mid in sorted(common_ids):
        old_ver = extract_version(old_names.get(mid, ""))
        new_ver = extract_version(new_names.get(mid, ""))
        if old_ver and new_ver and old_ver != new_ver:
            version_bumps.append({
                "model_id": mid,
                "display_name": new_names.get(mid, ""),
                "from": old_ver,
                "to": new_ver,
            })

    # --- Price changes ---
    price_changes = []
    for mid in sorted(common_ids):
        old_p = old_prices.get(mid, {})
        new_p = new_prices.get(mid, {})
        for field in ("input", "output", "cached"):
            old_v = old_p.get(field)
            new_v = new_p.get(field)
            if old_v is None or new_v is None or old_v == 0:
                continue
            change_pct = abs(new_v - old_v) / old_v
            if change_pct >= price_threshold:
                price_changes.append({
                    "model_id": mid,
                    "display_name": new_names.get(mid, ""),
                    "field": f"pricing_{field}_per_m",
                    "from": old_v,
                    "to": new_v,
                    "change_pct": round(change_pct * 100, 1),
                })

    # --- Rank changes ---
    rank_changes = []
    for mid in sorted(common_ids):
        old_rank = old_ranks.get(mid)
        new_rank = new_ranks.get(mid)
        if old_rank and new_rank:
            delta = old_rank - new_rank  # positive = moved up
            if abs(delta) >= rank_threshold:
                rank_changes.append({
                    "model_id": mid,
                    "display_name": new_names.get(mid, ""),
                    "from": old_rank,
                    "to": new_rank,
                    "delta": delta,
                    "direction": "up" if delta > 0 else "down",
                })

    has_changes = bool(new_models or version_bumps or price_changes or rank_changes)

    return {
        "check_date": dt.date.today().isoformat(),
        "has_changes": has_changes,
        "summary": {
            "new_models_count": len(new_models),
            "version_bumps_count": len(version_bumps),
            "price_changes_count": len(price_changes),
            "rank_changes_count": len(rank_changes),
        },
        "new_models": new_models,
        "version_bumps": version_bumps,
        "price_changes": price_changes,
        "rank_changes": rank_changes,
        "old_snapshot_id": old_snap.get("snapshot_id"),
        "new_snapshot_id": new_snap.get("snapshot_id"),
    }


def main():
    ap = argparse.ArgumentParser(description="Detect new models and changes between snapshots.")
    ap.add_argument("--snapshot-dir", help="Directory containing snapshot JSON files")
    ap.add_argument("--old", help="Path to older snapshot JSON")
    ap.add_argument("--new", help="Path to newer snapshot JSON")
    ap.add_argument("--out", required=True, help="Output change summary JSON path")
    ap.add_argument("--threshold-price", type=float, default=PRICE_CHANGE_THRESHOLD,
                    help=f"Price change threshold (fraction, default: {PRICE_CHANGE_THRESHOLD})")
    ap.add_argument("--threshold-rank", type=int, default=RANK_CHANGE_THRESHOLD,
                    help=f"Rank change threshold (positions, default: {RANK_CHANGE_THRESHOLD})")
    args = ap.parse_args()

    # Resolve old/new snapshot paths
    if args.old and args.new:
        old_path, new_path = args.old, args.new
    elif args.snapshot_dir:
        latest = find_latest_snapshots(args.snapshot_dir, count=2)
        if len(latest) < 2:
            print("[warn] fewer than 2 snapshots found; nothing to compare", file=sys.stderr)
            # Emit empty result
            result = {
                "check_date": dt.date.today().isoformat(),
                "has_changes": False,
                "summary": {"new_models_count": 0, "version_bumps_count": 0,
                            "price_changes_count": 0, "rank_changes_count": 0},
                "new_models": [], "version_bumps": [],
                "price_changes": [], "rank_changes": [],
                "old_snapshot_id": None, "new_snapshot_id": None,
            }
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[done] no previous snapshot; wrote empty result to {args.out}", file=sys.stderr)
            return
        old_path, new_path = latest
    else:
        ap.error("either --snapshot-dir or both --old and --new must be provided")

    print(f"[info] comparing {os.path.basename(old_path)} -> {os.path.basename(new_path)}", file=sys.stderr)

    old_snap = load_snapshot(old_path)
    new_snap = load_snapshot(new_path)

    result = detect_changes(old_snap, new_snap,
                            price_threshold=args.threshold_price,
                            rank_threshold=args.threshold_rank)

    # Annotate the new snapshot with previous snapshot ID
    new_snap["previous_snapshot_id"] = old_snap.get("snapshot_id")
    with open(new_path, "w", encoding="utf-8") as f:
        json.dump(new_snap, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Print human-readable summary
    s = result["summary"]
    print(f"[done] {s['new_models_count']} new, {s['version_bumps_count']} version bumps, "
          f"{s['price_changes_count']} price changes, {s['rank_changes_count']} rank shifts",
          file=sys.stderr)
    if result["has_changes"]:
        print("[info] changes detected — analysis pending", file=sys.stderr)
    else:
        print("[info] no significant changes", file=sys.stderr)


if __name__ == "__main__":
    main()
