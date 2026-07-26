#!/usr/bin/env python3
"""
Offline + agent eval suite for the NBA Insights API.

Metrics
-------
- tool_call_success: fraction of tool calls with ok=True
- tool_used: whether at least one tool was called
- faithfulness: fraction of gold must_include facts found in answer/citations
- expected_tool_hit: whether any expected tool appeared in the trace

Usage
-----
  cd ~/Desktop/crewai_nba_analysis
  python evals/run_eval.py                 # full agent eval (needs OPENAI_API_KEY)
  python evals/run_eval.py --limit 2       # smoke subset
  python evals/run_eval.py --tools-only    # no LLM; checks structured tools + gold facts
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_PATH = Path(__file__).with_name("dataset.json")
RESULTS_PATH = Path(__file__).with_name("results.json")


def _load_dataset() -> List[Dict[str, Any]]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _evidence_text(answer: str, citations: List[Dict[str, Any]]) -> str:
    parts = [answer or ""]
    for cite in citations or []:
        parts.append(json.dumps(cite.get("snippet") or cite, default=str))
    return "\n".join(parts)


def fact_in_text(text: str, fact: str) -> bool:
    """Case-insensitive substring match; numbers also match nearby decimals."""
    hay = text.casefold()
    needle = str(fact).casefold()
    if needle in hay:
        return True
    # numeric flexibility: "24" matches 24.46; "60" matches 60.0
    if re.fullmatch(r"-?\d+(\.\d+)?", str(fact).strip()):
        pattern = rf"(?<![\d.]){re.escape(str(fact).strip())}(?:\.\d+)?(?![\d])"
        return re.search(pattern, text) is not None
    return False


def score_case(
    case: Dict[str, Any],
    answer: str,
    citations: List[Dict[str, Any]],
    tool_trace: List[Dict[str, Any]],
) -> Dict[str, Any]:
    evidence = _evidence_text(answer, citations)
    must = case.get("must_include") or []
    hits = [f for f in must if fact_in_text(evidence, f)]
    misses = [f for f in must if f not in hits]

    tools = [t.get("tool") for t in tool_trace]
    ok_flags = [bool(t.get("ok", True)) for t in tool_trace]
    expected_any = case.get("expected_tools_any") or []
    expected_tool_hit = (not expected_any) or any(t in tools for t in expected_any)

    faithfulness = (len(hits) / len(must)) if must else 1.0
    tool_call_success = (sum(ok_flags) / len(ok_flags)) if ok_flags else 0.0

    return {
        "faithfulness": faithfulness,
        "faithfulness_hits": hits,
        "faithfulness_misses": misses,
        "tool_used": len(tool_trace) > 0,
        "tool_call_success": tool_call_success,
        "tool_count": len(tool_trace),
        "expected_tool_hit": expected_tool_hit,
        "tools_seen": tools,
        "passed": faithfulness >= 1.0 and expected_tool_hit and len(tool_trace) > 0 and tool_call_success == 1.0,
    }


def run_tools_only(cases: List[Dict[str, Any]], csv_path: str) -> Dict[str, Any]:
    """Validate structured tools return gold facts without calling the LLM."""
    from crewai_utils import get_agent_tools

    tools = {t.name: t for t in get_agent_tools(csv_path)}
    results = []
    started = time.perf_counter()

    for case in cases:
        # Probe the most relevant tools for grounding evidence
        blobs = []
        trace = []
        probes = [
            ("get_nba_data_summary", {}),
            ("analyze_nba_data", {"pandas_code": "df.nlargest(1, 'PTS')[['Player','PTS','Tm','Opp','Data']]"}),
            ("analyze_nba_data", {"pandas_code": "df.groupby('Player')['3P'].sum().sort_values(ascending=False).head(3)"}),
            ("analyze_nba_data", {"pandas_code": "df.groupby('Player')['AST'].sum().sort_values(ascending=False).head(3)"}),
            ("analyze_nba_data", {"pandas_code": "df[df['Player'].str.contains('LeBron', case=False, na=False)]['PTS'].mean()"}),
            ("analyze_nba_data", {"pandas_code": "pd.Series({'unique_players': df['Player'].nunique(), 'unique_teams': df['Tm'].nunique()})"}),
            ("search_nba_data", {"query": "Giannis", "column": "PTS", "value": "59", "limit": 5}),
            ("analyze_nba_data", {"pandas_code": "df.nlargest(1, 'TRB')[['Player','TRB','Opp','Data']]"}),
            ("analyze_nba_data", {"pandas_code": "df[df['Player'].str.contains('Joki', case=False, na=False)]['AST'].sum()"}),
            ("search_nba_data", {"query": "Fox", "column": "PTS", "value": "60", "limit": 5}),
            ("lookup_player_games", {"player": "Giannis", "pts": 59}),
            ("lookup_player_games", {"player": "DeAaron Fox", "pts": 60}),
        ]
        for name, kwargs in probes:
            raw = tools[name].run(**kwargs) if kwargs else tools[name].run()
            payload = json.loads(raw)
            blobs.append(raw)
            trace.append({
                "tool": name,
                "ok": bool(payload.get("ok", True)),
                "summary": payload.get("error") or payload.get("result_type") or "ok",
            })

        evidence = "\n".join(blobs)
        scored = score_case(case, evidence, [], trace)
        # tools-only: expected_tool_hit against probes is always true if any expected in probes
        results.append({
            "id": case["id"],
            "question": case["question"],
            "mode": "tools_only",
            "answer_preview": evidence[:500],
            **scored,
        })

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return _aggregate(results, mode="tools_only", elapsed_ms=elapsed_ms)


def run_agent_eval(cases: List[Dict[str, Any]], csv_path: str) -> Dict[str, Any]:
    from crewai_utils import run_ask

    results = []
    started = time.perf_counter()
    for case in cases:
        print(f"→ {case['id']}: {case['question'][:80]}...")
        try:
            out = run_ask(case["question"], csv_path=csv_path)
            scored = score_case(
                case,
                out.get("answer", ""),
                out.get("citations") or [],
                out.get("tool_trace") or [],
            )
            results.append({
                "id": case["id"],
                "question": case["question"],
                "mode": "agent",
                "answer": out.get("answer"),
                "citations": out.get("citations"),
                "tool_trace": out.get("tool_trace"),
                "latency_ms": out.get("latency_ms"),
                **scored,
            })
            status = "PASS" if scored["passed"] else "FAIL"
            print(f"  {status} faithfulness={scored['faithfulness']:.2f} tools={scored['tools_seen']}")
        except Exception as e:
            results.append({
                "id": case["id"],
                "question": case["question"],
                "mode": "agent",
                "error": str(e),
                "faithfulness": 0.0,
                "tool_used": False,
                "tool_call_success": 0.0,
                "expected_tool_hit": False,
                "passed": False,
            })
            print(f"  ERROR {e}")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return _aggregate(results, mode="agent", elapsed_ms=elapsed_ms)


def _aggregate(results: List[Dict[str, Any]], mode: str, elapsed_ms: int) -> Dict[str, Any]:
    n = len(results) or 1
    latencies = [r["latency_ms"] for r in results if isinstance(r.get("latency_ms"), (int, float))]
    summary = {
        "mode": mode,
        "n_cases": len(results),
        "pass_rate": sum(1 for r in results if r.get("passed")) / n,
        "mean_faithfulness": sum(r.get("faithfulness", 0.0) for r in results) / n,
        "tool_used_rate": sum(1 for r in results if r.get("tool_used")) / n,
        "mean_tool_call_success": sum(r.get("tool_call_success", 0.0) for r in results) / n,
        "expected_tool_hit_rate": sum(1 for r in results if r.get("expected_tool_hit")) / n,
        "latency_ms_mean": statistics.mean(latencies) if latencies else None,
        "latency_ms_p50": statistics.median(latencies) if latencies else None,
        "total_elapsed_ms": elapsed_ms,
        "cases": results,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="NBA Insights eval suite")
    parser.add_argument("--limit", type=int, default=None, help="Run only first N cases")
    parser.add_argument("--tools-only", action="store_true", help="Skip LLM; score tool JSON against gold facts")
    parser.add_argument("--csv", default=str(ROOT / "nba24-25.csv"), help="Path to NBA CSV")
    parser.add_argument("--out", default=str(RESULTS_PATH), help="Where to write results JSON")
    args = parser.parse_args()

    cases = _load_dataset()
    if args.limit is not None:
        cases = cases[: args.limit]

    if args.tools_only:
        report = run_tools_only(cases, args.csv)
    else:
        report = run_agent_eval(cases, args.csv)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Eval summary ===")
    for key in (
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
        print(f"{key}: {report.get(key)}")
    print(f"wrote: {out_path}")
    return 0 if report.get("pass_rate", 0) == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
