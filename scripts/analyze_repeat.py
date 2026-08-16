#!/usr/bin/env python3
"""
Run-to-run variance analysis.

Compares repeated runs of the same scenario, executed with a server restart
between them, and separates two sources of variability:

  within-run  : spread across requests inside a single run
  across-run  : spread of per-run medians between runs

Across-run variance is usually the larger of the two and is what a reader
should treat as the real uncertainty on a reported number. Most benchmarks
report only the within-run figure, which understates it.

Usage:
    python scripts/analyze_repeat.py --scenario T1rep
    python scripts/analyze_repeat.py --runs id1,id2,id3
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

RAW_DIR = Path("results/raw")
PROC_DIR = Path("results/processed")


def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    return s[max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))]


def cv(values):
    """Coefficient of variation, percent."""
    if len(values) < 2:
        return 0.0
    m = statistics.mean(values)
    return statistics.stdev(values) / m * 100 if m else float("nan")


def discover(scenario):
    out = []
    for d in sorted(RAW_DIR.iterdir()):
        mf = d / "manifest.json" if d.is_dir() else None
        if not mf or not mf.exists():
            continue
        try:
            man = json.loads(mf.read_text())
        except Exception:
            continue
        if man.get("scenario") == scenario:
            out.append((d.name, man))
    return out


def load_cells(run_id):
    """Return {(isl, osl): {metric: value}} for one run."""
    cells = {}
    with (RAW_DIR / run_id / "requests.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            cells.setdefault((r["isl_target"], r["osl_target"]), []).append(r)

    out = {}
    for key, recs in cells.items():
        ttfts = [r["ttft_ms"] for r in recs]
        tpots = [statistics.mean(r["itls_ms"]) for r in recs if r["itls_ms"]]
        pooled = [x for r in recs for x in r["itls_ms"]]
        out[key] = {
            "n": len(recs),
            "isl_actual": statistics.median([r["prompt_tokens"] for r in recs]),
            "ttft_p50": pct(ttfts, 50),
            "ttft_p95": pct(ttfts, 95),
            "tpot_mean": statistics.mean(tpots),
            "speed_tok_s": 1000 / statistics.mean(tpots),
            "itl_p99": pct(pooled, 99),
            "within_cv_tpot": cv(tpots),
            "within_cv_ttft": cv(ttfts),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="T1rep")
    ap.add_argument("--runs", default=None, help="comma-separated run ids")
    ap.add_argument("--label", default="t1_repeatability")
    args = ap.parse_args()

    if args.runs:
        run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
        manifests = {r: json.loads((RAW_DIR / r / "manifest.json").read_text())
                     for r in run_ids}
    else:
        found = discover(args.scenario)
        if len(found) < 2:
            raise SystemExit(f"Need at least 2 runs of scenario {args.scenario!r}; "
                             f"found {len(found)}")
        run_ids = [r for r, _ in found]
        manifests = dict(found)

    print(f"Scenario : {args.scenario}")
    print(f"Runs     : {len(run_ids)}\n")
    for r in run_ids:
        m = manifests[r]
        g = m.get("gpu_idle_before") or m.get("gpu_before") or {}
        print(f"  {r}")
        print(f"    note   : {m.get('note')!r}")
        print(f"    commit : {m.get('git_commit')}")
        print(f"    GPU at start: {g.get('temperature_c')}C, "
              f"{g.get('memory_used_mib')} MiB used")

    # Config sanity: all runs should share the same code and packages.
    commits = {manifests[r].get("git_commit") for r in run_ids}
    if len(commits) > 1:
        print(f"\n  [!] Runs span multiple commits: {commits}")
    pkgs = {json.dumps(manifests[r].get("packages"), sort_keys=True) for r in run_ids}
    if len(pkgs) > 1:
        print("  [!] Package versions differ between runs")

    per_run = {r: load_cells(r) for r in run_ids}
    keys = sorted(set.intersection(*[set(v) for v in per_run.values()]))
    if not keys:
        raise SystemExit("No cells common to all runs")

    rows = []
    for key in keys:
        isl, osl = key
        entry = {"isl_target": isl, "osl_target": osl,
                 "isl_actual": per_run[run_ids[0]][key]["isl_actual"]}
        for metric in ("ttft_p50", "tpot_mean", "speed_tok_s", "itl_p99"):
            vals = [per_run[r][key][metric] for r in run_ids]
            entry[f"{metric}_mean"] = statistics.mean(vals)
            entry[f"{metric}_min"] = min(vals)
            entry[f"{metric}_max"] = max(vals)
            entry[f"{metric}_across_cv"] = cv(vals)
            entry[f"{metric}_runs"] = vals
        entry["within_cv_tpot"] = statistics.mean(
            [per_run[r][key]["within_cv_tpot"] for r in run_ids])
        entry["within_cv_ttft"] = statistics.mean(
            [per_run[r][key]["within_cv_ttft"] for r in run_ids])
        rows.append(entry)

    # --- Per-cell detail ---
    print("\n=== Decode speed (tok/s) per run ===")
    hdr = f"{'ISL':>7}{'OSL':>6}" + "".join(f"{'run'+str(i+1):>10}" for i in range(len(run_ids)))
    hdr += f"{'mean':>10}{'range':>9}{'across CV':>11}{'within CV':>11}"
    print(hdr)
    print("-" * len(hdr))
    for e in rows:
        line = f"{e['isl_actual']:>7.0f}{e['osl_target']:>6}"
        line += "".join(f"{v:>10.2f}" for v in e["speed_tok_s_runs"])
        rng = e["speed_tok_s_max"] - e["speed_tok_s_min"]
        line += (f"{e['speed_tok_s_mean']:>10.2f}{rng:>9.2f}"
                 f"{e['speed_tok_s_across_cv']:>10.3f}%{e['within_cv_tpot']:>10.3f}%")
        print(line)

    print("\n=== TTFT p50 (ms) per run ===")
    hdr = f"{'ISL':>7}{'OSL':>6}" + "".join(f"{'run'+str(i+1):>10}" for i in range(len(run_ids)))
    hdr += f"{'mean':>10}{'range':>9}{'across CV':>11}"
    print(hdr)
    print("-" * len(hdr))
    for e in rows:
        line = f"{e['isl_actual']:>7.0f}{e['osl_target']:>6}"
        line += "".join(f"{v:>10.2f}" for v in e["ttft_p50_runs"])
        rng = e["ttft_p50_max"] - e["ttft_p50_min"]
        line += (f"{e['ttft_p50_mean']:>10.2f}{rng:>9.2f}"
                 f"{e['ttft_p50_across_cv']:>10.3f}%")
        print(line)

    # --- Verdict ---
    print("\n=== Variance summary ===")
    for metric, label in (("speed_tok_s", "decode speed"),
                          ("ttft_p50", "TTFT p50"),
                          ("itl_p99", "ITL p99")):
        across = [e[f"{metric}_across_cv"] for e in rows]
        print(f"{label:<14} across-run CV: "
              f"median {statistics.median(across):.3f}%  max {max(across):.3f}%")

    w_tpot = statistics.median([e["within_cv_tpot"] for e in rows])
    a_speed = statistics.median([e["speed_tok_s_across_cv"] for e in rows])
    print(f"\nwithin-run CV (decode) : {w_tpot:.3f}%")
    print(f"across-run CV (decode) : {a_speed:.3f}%")
    if w_tpot > 0:
        ratio = a_speed / w_tpot
        print(f"ratio                  : {ratio:.1f}x")
        if ratio > 3:
            print("  -> Server restart dominates the uncertainty. Report the")
            print("     across-run figure as the error bar on decode speed.")
        elif ratio > 1.2:
            print("  -> Across-run variance is somewhat larger, as expected.")
        else:
            print("  -> Both sources are comparable; the setup is unusually stable.")

    # --- CSV ---
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PROC_DIR / f"{args.label}.csv"
    flat = []
    for e in rows:
        row = {k: v for k, v in e.items() if not k.endswith("_runs")}
        for metric in ("ttft_p50", "tpot_mean", "speed_tok_s", "itl_p99"):
            for i, v in enumerate(e[f"{metric}_runs"]):
                row[f"{metric}_run{i+1}"] = v
        flat.append(row)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    main()
