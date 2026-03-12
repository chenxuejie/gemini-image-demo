"""
Gemini 3 Flash Preview Latency Test — Google GenAI SDK
Uses google.genai Client directly with native thinking_level support.

Measures:
  - TTFT (Time to First Token) via streaming API
  - Total Latency via streaming API (time until last token)
  - Thinking tokens, output tokens, TPS

Thinking levels for Gemini 3 Flash:
  MINIMAL — as close to zero thinking as possible (Flash only)
  LOW     — fewer thinking tokens, for simpler tasks
  MEDIUM  — balanced approach (Flash only)
  HIGH    — default, deep reasoning
"""

import time
import statistics
import csv
import os

from google import genai
from google.genai import types

# ─── Configuration ───
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LOCATION = "global"  # Gemini 3 models require 'global'
MODEL_ID = "gemini-3-flash-preview"
NUM_ITERATIONS = 3

THINKING_LEVELS = [
    types.ThinkingLevel.MINIMAL,
    types.ThinkingLevel.LOW,
    types.ThinkingLevel.MEDIUM,
    types.ThinkingLevel.HIGH,
]

TEST_PROMPT = (
    "Explain the key differences between supervised learning and unsupervised learning "
    "in machine learning. Provide 2 examples for each. Keep your answer under 200 words."
)

# ─── Client ───
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


# ─── Helpers ───

def test_streaming(thinking_level, prompt: str, debug: bool = False) -> dict:
    """
    Call generate_content_stream to measure:
      - TTFT: time from request start to first text chunk
      - Total latency: time from request start to last chunk
      - Token counts from usage_metadata
    """
    config = types.GenerateContentConfig(
        max_output_tokens=1024,
        temperature=0.4,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
    )

    text_chunks = []
    ttft = None
    usage_metadata = None

    start = time.perf_counter()

    response_stream = client.models.generate_content_stream(
        model=MODEL_ID,
        contents=prompt,
        config=config,
    )

    for chunk in response_stream:
        now = time.perf_counter()

        # Check if this chunk has text content
        has_text = False
        if hasattr(chunk, "text") and chunk.text:
            has_text = True
            text_chunks.append(chunk.text)

        # Record TTFT on the first chunk that has actual text
        if ttft is None and has_text:
            ttft = now - start

        # Capture usage_metadata (usually on the last chunk)
        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            usage_metadata = chunk.usage_metadata

    total_latency = time.perf_counter() - start

    # If no text chunks arrived, TTFT = total latency
    if ttft is None:
        ttft = total_latency

    # Extract token counts
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    thinking_tokens = 0

    if usage_metadata:
        input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage_metadata, "total_token_count", 0) or 0
        thinking_tokens = getattr(usage_metadata, "thoughts_token_count", 0) or 0

    if debug:
        print(
            f"    🔍 in={input_tokens}  out={output_tokens}  "
            f"thinking={thinking_tokens}  total={total_tokens}"
        )

    full_text = "".join(text_chunks)
    tps = round(output_tokens / total_latency, 1) if total_latency > 0 and output_tokens else 0

    return {
        "ttft": round(ttft, 3),
        "total_latency": round(total_latency, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "thinking_tokens": thinking_tokens,
        "tps": tps,
        "preview": (full_text[:100] + "...") if len(full_text) > 100 else full_text,
    }


def fmt(val, suffix=""):
    if val is None:
        return "N/A"
    return f"{val:.3f}{suffix}" if isinstance(val, float) else f"{val}{suffix}"


def sep(char="=", w=100):
    print(char * w)


# ─── Main ───

def main():
    level_names = [lv.name for lv in THINKING_LEVELS]

    sep()
    print("  Gemini 3 Flash Preview — TTFT & Total Latency vs Thinking Level (google.genai SDK)")
    print(f"  Model: {MODEL_ID} | Project: {PROJECT_ID} | Location: {LOCATION}")
    print(f"  Iterations: {NUM_ITERATIONS} per level | Method: streaming")
    print(f"  Levels: {level_names}")
    sep()
    print(f"  Prompt: {TEST_PROMPT[:70]}...")
    sep()
    print()

    results = []

    for level in THINKING_LEVELS:
        name = level.name
        sep("-")
        print(f"🧠 thinking_level = {name}")
        sep("-")

        print(f"\n  ▶ generate_content_stream() — {NUM_ITERATIONS} iterations")
        ttfts, total_lats, tps_list = [], [], []
        out_list, think_list, total_tok_list = [], [], []
        preview = None

        for i in range(NUM_ITERATIONS):
            try:
                r = test_streaming(level, TEST_PROMPT, debug=(i == 0))
                ttfts.append(r["ttft"])
                total_lats.append(r["total_latency"])
                tps_list.append(r["tps"])
                out_list.append(r["output_tokens"])
                think_list.append(r["thinking_tokens"])
                total_tok_list.append(r["total_tokens"])
                if preview is None:
                    preview = r["preview"]
                print(
                    f"    [{i+1}/{NUM_ITERATIONS}] "
                    f"TTFT={r['ttft']:.3f}s  "
                    f"total={r['total_latency']:.3f}s  "
                    f"out={r['output_tokens']}  "
                    f"thinking={r['thinking_tokens']}  "
                    f"tps={r['tps']}"
                )
            except Exception as e:
                print(f"    [{i+1}/{NUM_ITERATIONS}] ERROR: {e}")

        if preview:
            print(f"    Response: {preview}")

        row = {
            "level": name,
            "ttft_mean": statistics.mean(ttfts) if ttfts else None,
            "ttft_min": min(ttfts) if ttfts else None,
            "ttft_max": max(ttfts) if ttfts else None,
            "lat_mean": statistics.mean(total_lats) if total_lats else None,
            "lat_min": min(total_lats) if total_lats else None,
            "lat_max": max(total_lats) if total_lats else None,
            "out_mean": statistics.mean(out_list) if out_list else None,
            "think_mean": statistics.mean(think_list) if think_list else None,
            "total_tok_mean": statistics.mean(total_tok_list) if total_tok_list else None,
            "tps_mean": (
                statistics.mean([t for t in tps_list if t > 0])
                if any(t > 0 for t in tps_list)
                else None
            ),
            "error": None,
        }
        print(
            f"\n  📊 {name}: TTFT={fmt(row['ttft_mean'],'s')}  "
            f"total={fmt(row['lat_mean'],'s')}  "
            f"think={fmt(row['think_mean'])}  tps={fmt(row['tps_mean'])}"
        )
        print()
        results.append(row)

    # ═══ Summary ═══
    sep("=")
    print("  📋 FINAL SUMMARY — TTFT & Total Latency by Thinking Level (google.genai)")
    sep("=")
    hdr = (
        f"{'Level':>8} | "
        f"{'TTFT Mean':>9} {'TTFT Min':>8} {'TTFT Max':>8} | "
        f"{'Lat Mean':>9} {'Lat Min':>8} {'Lat Max':>8} | "
        f"{'Out':>5} {'Think':>6} {'Total':>6} | "
        f"{'TPS':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r.get("error"):
            print(f"{r['level']:>8} | ERROR: {r['error'][:60]}")
            continue
        print(
            f"{r['level']:>8} | "
            f"{fmt(r['ttft_mean'],'s'):>9} {fmt(r['ttft_min'],'s'):>8} {fmt(r['ttft_max'],'s'):>8} | "
            f"{fmt(r['lat_mean'],'s'):>9} {fmt(r['lat_min'],'s'):>8} {fmt(r['lat_max'],'s'):>8} | "
            f"{fmt(r['out_mean']):>5} {fmt(r['think_mean']):>6} {fmt(r['total_tok_mean']):>6} | "
            f"{fmt(r['tps_mean']):>5}"
        )
    sep("=")

    # CSV
    csv_path = os.path.join(os.path.dirname(__file__), "latency_test_genai_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "ttft_mean", "ttft_min", "ttft_max",
                "lat_mean", "lat_min", "lat_max",
                "out_mean", "think_mean", "total_tok_mean",
                "tps_mean", "error",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  💾 Saved to: {csv_path}")

    sep("=")
    print("  ✅ Complete!")
    sep("=")


if __name__ == "__main__":
    main()
