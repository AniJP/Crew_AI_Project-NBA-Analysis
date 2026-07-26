#!/usr/bin/env python3
"""
NBA Insights eval runner via the FastAPI server.

Why this exists:
- Importing `crewai` in some environments can crash Python (native segfault).
- The FastAPI server already has the working agent loaded, so we call `POST /ask`
  and score using only stdlib (json/re/urllib).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(__file__).with_name("dataset.json")
RESULTS_PATH = Path(__file__).with_name("results.json")


def _fact_in_text(text: str, fact: str) -> bool:
    hay = (text or "").casefold()
    needle = str(fact).strip().casefold()
    if not needle:
        return True
    if needle and needle in hay:
        return True

    # Numeric flexibility: allow matching "60" against "60.0" and vice versa.
    if re.fullmatch(r"-?\d+(\.\d+)?", str(fact).strip()):
        s = str(fact).strip()
        # Match as a token-ish number boundary to reduce false positives.
        pattern = rf"(?<![0-9.]){re.escape(s)}(?:[0-9.])?(?![0-9.])"
        return re.search(pattern, hay) is not None

    return False


def _evidence_text(answer: str, citations: List[Dict[str, Any]]) -> str:
    parts = [answer or ""]
    for c in citations or []:
        snippet = c.get("snippet")
        if snippet is not None:
            parts.append(json.dumps(snippet, default=str))
        else:
            parts.append(json.dumps(c, default=str))
    return "\n".join(parts)


def _score_case(
    case: Dict[str, Any],
    api_response: Dict[str, Any],
) -> Dict[str, Any]:
    answer = api_response.get("answer") or ""
    citations = api_response.get("citations") or []
    tool_trace = api_response.get("tool_trace") or []

    evidence = _evidence_text(answer, citations)
    must = case.get("must_include") or []
    hits = [f for f in must if _fact_in_text(evidence, f)]
    misses = [f for f in must if f not in hits]

    expected_any = case.get("expected_tools_any") or []
    seen_tools = [t.get("tool") for t in tool_trace if isinstance(t, dict)]
    expected_tool_hit = (not expected_any) or any(t in seen_tools for t in expected_any)

    tool_calls_ok = [bool(t.get("ok", True)) for t in tool_trace if isinstance(t, dict)]
    tool_call_success = (sum(tool_calls_ok) / len(tool_calls_ok)) if tool_calls_ok else 0.0
    any_tool_ok = any(tool_calls_ok)

    faithfulness = (len(hits) / len(must)) if must else 1.0

    # Pass if the answer is fully grounded and at least one tool succeeded.
    # A failed exploratory tool (e.g. broken semantic search) should not fail the case.
    return {
        "faithfulness": faithfulness,
        "faithfulness_hits": hits,
        "faithfulness_misses": misses,
        "tool_used": len(tool_trace) > 0,
        "tool_call_success": tool_call_success,
        "any_tool_ok": any_tool_ok,
        "tool_count": len(tool_trace),
        "expected_tool_hit": expected_tool_hit,
        "tools_seen": seen_tools,
        "passed": (
            faithfulness >= 1.0
            and expected_tool_hit
            and any_tool_ok
        ),
    }


def _post_json(url: str, payload: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="NBA eval via FastAPI /ask")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N cases")
    parser.add_argument("--timeout-s", type=int, default=120, help="Request timeout per case")
    args = parser.parse_args()

    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if args.limit is not None:
        cases = cases[: args.limit]

    url = args.base_url.rstrip("/") + "/ask"
    results: List[Dict[str, Any]] = []

    started = time.perf_counter()
    for case in cases:
        cid = case.get("id", "unknown")
        q = case["question"]
        print(f"→ {cid}: {q[:80]}...")
        try:
            resp = _post_json(url, {"question": q}, timeout_s=args.timeout_s)
            scored = _score_case(case, resp)
            results.append(
                {
                    "id": cid,
                    "question": q,
                    "mode": "api",
                    "answer": resp.get("answer"),
                    "citations": resp.get("citations") or [],
                    "tool_trace": resp.get("tool_trace") or [],
                    "latency_ms": resp.get("latency_ms"),
                    **scored,
                }
            )
            status = "PASS" if scored["passed"] else "FAIL"
            print(
                f"  {status} faithfulness={scored['faithfulness']:.2f} "
                f"tools={scored['tool_count']} expected_tool_hit={scored['expected_tool_hit']}"
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
            results.append(
                {
                    "id": cid,
                    "question": q,
                    "mode": "api",
                    "error": f"HTTPError {e.code}: {body[:500]}",
                    "faithfulness": 0.0,
                    "tool_used": False,
                    "tool_call_success": 0.0,
                    "expected_tool_hit": False,
                    "passed": False,
                }
            )
            print(f"  ERROR HTTP {e.code}")
        except Exception as e:
            results.append(
                {
                    "id": cid,
                    "question": q,
                    "mode": "api",
                    "error": str(e),
                    "faithfulness": 0.0,
                    "tool_used": False,
                    "tool_call_success": 0.0,
                    "expected_tool_hit": False,
                    "passed": False,
                }
            )
            print(f"  ERROR {type(e).__name__}: {e}")

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    latencies = [r.get("latency_ms") for r in results if isinstance(r.get("latency_ms"), int)]
    summary = {
        "mode": "api",
        "n_cases": len(results),
        "pass_rate": (sum(1 for r in results if r.get("passed")) / (len(results) or 1)),
        "mean_faithfulness": (sum(r.get("faithfulness", 0.0) for r in results) / (len(results) or 1)),
        "tool_used_rate": (sum(1 for r in results if r.get("tool_used")) / (len(results) or 1)),
        "mean_tool_call_success": (sum(r.get("tool_call_success", 0.0) for r in results) / (len(results) or 1)),
        "expected_tool_hit_rate": (sum(1 for r in results if r.get("expected_tool_hit")) / (len(results) or 1)),
        "latency_ms_mean": statistics.mean(latencies) if latencies else None,
        "latency_ms_p50": statistics.median(latencies) if latencies else None,
        "total_elapsed_ms": elapsed_ms,
        "cases": results,
    }

    RESULTS_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Eval summary ===")
    for k in (
        "mode",
        "n_cases",
        "pass_rate",
        "mean_faithfulness",
        "tool_used_rate",
        "mean_tool_call_success",
        "expected_tool_hit_rate",
        "latency_ms_mean",
        "latency_ms_p50",
        "total_elapsed_ms",
    ):
        print(f"{k}: {summary.get(k)}")
    print(f"wrote: {RESULTS_PATH}")

    return 0 if summary.get("pass_rate", 0) == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

