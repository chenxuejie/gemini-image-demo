"""
Gemini 3 Flash Preview Latency Test
Uses ChatVertexAI from LangChain to measure latency metrics
across different thinking levels.

Gemini 3 models use thinking_level (MINIMAL, LOW, MEDIUM, HIGH).
Since the current google-cloud-aiplatform SDK (v1.134.0) doesn't support
thinking_level in ThinkingConfig proto yet, we map thinking levels to
equivalent thinking_budget values:
  MINIMAL → 0    (as close as possible to zero budget)
  LOW     → 1024 (fewer tokens for simpler tasks)
  MEDIUM  → 8192 (balanced approach)
  HIGH    → 24576 (default, deep reasoning)
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import statistics
import csv
import os
import json
from datetime import datetime

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = "global"
MODEL_ID = "gemini-3-flash-preview"
NUM_ITERATIONS = 3

# Thinking levels mapped to thinking_budget equivalents
THINKING_LEVELS = {
    "MINIMAL": 0,       # as close to zero as possible
    "LOW": 1024,        # fewer tokens, simpler tasks
    "MEDIUM": 8192,     # balanced
    "HIGH": 24576,      # default, deep reasoning
}

# Test prompt
TEST_PROMPT = (
    "Explain the key differences between supervised learning and unsupervised learning "
    "in machine learning. Provide 2 examples for each. Keep your answer under 200 words."
)


def extract_text(content):
    """Extract plain text from response content (handles list or string)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item["text"] if isinstance(item, dict) and "text" in item
            else item if isinstance(item, str) else ""
            for item in content
        )
    return str(content)


def create_llm(thinking_budget: int):
    """Create ChatVertexAI with a specific thinking_budget."""
    return ChatVertexAI(
        model=MODEL_ID,
        project=PROJECT_ID,
        location=LOCATION,
        max_output_tokens=1024,
        temperature=0.4,
        thinking_budget=thinking_budget,
    )


def test_invoke(llm, prompt_text: str, debug: bool = False) -> dict:
    """Measure total latency using invoke."""
    message = HumanMessage(content=prompt_text)

    start = time.perf_counter()
    response = llm.invoke([message])
    total_time = time.perf_counter() - start

    if debug:
        resp_meta = getattr(response, "response_metadata", None) or {}
        if isinstance(resp_meta, dict) and "usage_metadata" in resp_meta:
            rm = resp_meta["usage_metadata"]
            print(f"    🔍 thoughts_token_count={rm.get('thoughts_token_count', 'N/A')}, "
                  f"total_token_count={rm.get('total_token_count', 'N/A')}")

    # Token info
    usage = getattr(response, "usage_metadata", None) or {}
    input_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
    output_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
    total_tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0

    # Thinking tokens
    thinking_tokens = 0
    resp_meta = getattr(response, "response_metadata", None) or {}
    if isinstance(resp_meta, dict) and "usage_metadata" in resp_meta:
        rm_usage = resp_meta["usage_metadata"]
        if isinstance(rm_usage, dict):
            thinking_tokens = rm_usage.get("thoughts_token_count", 0) or 0
    if thinking_tokens == 0 and isinstance(usage, dict):
        otd = usage.get("output_token_details", {})
        if isinstance(otd, dict):
            thinking_tokens = otd.get("reasoning", 0) or 0

    text = extract_text(getattr(response, "content", ""))

    return {
        "latency": round(total_time, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "thinking_tokens": thinking_tokens,
        "tps": round(output_tokens / total_time, 1) if total_time > 0 and output_tokens else 0,
        "preview": (text[:100] + "...") if len(text) > 100 else text,
    }


def fmt(val, suffix=""):
    if val is None:
        return "N/A"
    return f"{val:.3f}{suffix}" if isinstance(val, float) else f"{val}{suffix}"


def sep(char="=", w=90):
    print(char * w)


def main():
    sep()
    print(f"  Gemini 3 Flash Preview — Latency vs Thinking Level Test")
    print(f"  Model: {MODEL_ID} | Project: {PROJECT_ID} | Location: {LOCATION}")
    print(f"  Iterations: {NUM_ITERATIONS} per level")
    print(f"  Levels: {list(THINKING_LEVELS.keys())}")
    print(f"  Note: thinking_level mapped to thinking_budget (proto limitation)")
    sep()
    print(f"  Prompt: {TEST_PROMPT[:70]}...")
    sep()
    print()

    results = []

    for level_name, budget in THINKING_LEVELS.items():
        sep("-")
        print(f"🧠 {level_name} (thinking_budget={budget})")
        sep("-")

        try:
            llm = create_llm(budget)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({"level": level_name, "budget": budget, "error": str(e)})
            continue

        print(f"\n  ▶ invoke() — {NUM_ITERATIONS} iterations")
        lats, tps_list, out_list, think_list, total_list = [], [], [], [], []
        preview = None

        for i in range(NUM_ITERATIONS):
            try:
                r = test_invoke(llm, TEST_PROMPT, debug=(i == 0))
                lats.append(r["latency"])
                tps_list.append(r["tps"])
                out_list.append(r["output_tokens"])
                think_list.append(r["thinking_tokens"])
                total_list.append(r["total_tokens"])
                if preview is None:
                    preview = r["preview"]
                print(
                    f"    [{i+1}/{NUM_ITERATIONS}] "
                    f"latency={r['latency']:.3f}s  "
                    f"out={r['output_tokens']}  "
                    f"thinking={r['thinking_tokens']}  "
                    f"total={r['total_tokens']}  "
                    f"tps={r['tps']}"
                )
            except Exception as e:
                print(f"    [{i+1}/{NUM_ITERATIONS}] ERROR: {e}")

        if preview:
            print(f"    Response: {preview}")

        row = {
            "level": level_name,
            "budget": budget,
            "lat_mean": statistics.mean(lats) if lats else None,
            "lat_min": min(lats) if lats else None,
            "lat_max": max(lats) if lats else None,
            "out_mean": statistics.mean(out_list) if out_list else None,
            "think_mean": statistics.mean(think_list) if think_list else None,
            "total_mean": statistics.mean(total_list) if total_list else None,
            "tps_mean": statistics.mean([t for t in tps_list if t > 0]) if any(t > 0 for t in tps_list) else None,
            "error": None,
        }
        print(f"\n  📊 {level_name}: lat_mean={fmt(row['lat_mean'],'s')}  think={fmt(row['think_mean'])}  tps={fmt(row['tps_mean'])}")
        print()
        results.append(row)

    # ═══ SUMMARY ═══
    sep("=")
    print("  📋 FINAL SUMMARY — Latency by Thinking Level")
    sep("=")
    hdr = f"{'Level':>8} {'Budget':>7} | {'Lat Mean':>9} {'Lat Min':>8} {'Lat Max':>8} | {'Out Tok':>7} {'Think Tok':>9} {'Total Tok':>9} | {'TPS':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r.get("error"):
            print(f"{r['level']:>8} {r['budget']:>7} | ERROR: {r['error'][:40]}")
            continue
        print(
            f"{r['level']:>8} {r['budget']:>7} | "
            f"{fmt(r['lat_mean'],'s'):>9} {fmt(r['lat_min'],'s'):>8} {fmt(r['lat_max'],'s'):>8} | "
            f"{fmt(r['out_mean']):>7} {fmt(r['think_mean']):>9} {fmt(r['total_mean']):>9} | "
            f"{fmt(r['tps_mean']):>5}"
        )
    sep("=")

    # CSV
    csv_path = os.path.join(os.path.dirname(__file__), "latency_test_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "level", "budget", "lat_mean", "lat_min", "lat_max",
            "out_mean", "think_mean", "total_mean", "tps_mean", "error"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  💾 Saved to: {csv_path}")
    sep("=")
    print("  ✅ Complete!")
    sep("=")


if __name__ == "__main__":
    main()
