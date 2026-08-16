#!/usr/bin/env python3
"""
T2 analysis: concurrency saturation curve.

Reads summary.json from every T2 run under results/raw/, merges them by
concurrency level, prints a table, writes a CSV, and renders the saturation
curve plot.

Runs whose manifest note contains "split" are excluded by default: those are
client-bottleneck validation runs that each carry half the total load, so
mixing them into the curve would understate throughput at that level.

Usage:
    python scripts/analyze_t2.py
    python scripts/analyze_t2.py --plot
    python scripts/analyze_t2.py --runs 20260815T195213Z-cb17a9,20260815T201743Z-bb1e31
    python scripts/analyze_t2.py --plot --slo-ttft 1000 --slo-itl 20
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

RAW_DIR = Path("results/raw")
PROC_DIR = Path("results/processed")
PLOT_DIR = Path("results/plots")

# Fields averaged when the same concurrency level appears in several runs.
MERGE_FIELDS = [
    "sys_out_tok_s", "sys_total_tok_s", "req_per_s", "user_speed_tok_s",
    "efficiency", "ttft_p50", "ttft_p95", "ttft_p99",
    "e2e_p50", "e2e_p95", "tpot_mean", "itl_p50", "itl_p95", "itl_p99",
]


def discover_runs(include_split=False):
    """Find every T2 run directory that has a usable summary.json."""
    found = []
    for d in sorted(RAW_DIR.iterdir()):
        if not d.is_dir():
            continue
        mf, sf = d / "manifest.json", d / "summary.json"
        if not (mf.exists() and sf.exists()):
            continue
        try:
            man = json.loads(mf.read_text())
        except Exception:
            continue
        if man.get("scenario") != "T2":
            continue
        note = (man.get("note") or "").lower()
        if "split" in note and not include_split:
            continue
        found.append((d.name, man))
    return found


def load_summaries(run_ids):
    """Return {concurrency: [(run_id, summary_dict), ...]}."""
    by_conc = {}
    for rid in run_ids:
        summaries = json.loads((RAW_DIR / rid / "summary.json").read_text())
        for s in summaries:
            by_conc.setdefault(s["concurrency"], []).append((rid, s))
    return by_conc


def merge(by_conc):
    rows, dupes = [], []
    for conc in sorted(by_conc):
        entries = by_conc[conc]
        row = {"concurrency": conc, "n_runs": len(entries)}
        for f in MERGE_FIELDS:
            vals = [s[f] for _, s in entries if f in s]
            row[f] = statistics.mean(vals) if vals else float("nan")
        row["n_requests"] = sum(s.get("n_requests", 0) for _, s in entries)
        row["n_errors"] = sum(s.get("n_errors", 0) for _, s in entries)
        rows.append(row)
        if len(entries) > 1:
            spread = [s["sys_out_tok_s"] for _, s in entries]
            dupes.append((conc, [r for r, _ in entries],
                          (max(spread) - min(spread)) / statistics.mean(spread) * 100))
    return rows, dupes


def classify(rows, slo_ttft, slo_itl):
    """Highest concurrency level that still meets both SLO targets."""
    ok = [r for r in rows if r["ttft_p95"] <= slo_ttft and r["itl_p95"] <= slo_itl]
    return max(ok, key=lambda r: r["concurrency"]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="comma-separated run ids")
    ap.add_argument("--include-split", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--slo-ttft", type=float, default=1000.0, help="TTFT p95 budget, ms")
    ap.add_argument("--slo-itl", type=float, default=20.0, help="ITL p95 budget, ms")
    ap.add_argument("--label", default="t2", help="output filename stem")
    args = ap.parse_args()

    if args.runs:
        run_ids = [r.strip() for r in args.runs.split(",") if r.strip()]
        manifests = {r: json.loads((RAW_DIR / r / "manifest.json").read_text())
                     for r in run_ids}
    else:
        found = discover_runs(args.include_split)
        if not found:
            raise SystemExit("No T2 runs found under results/raw/")
        run_ids = [r for r, _ in found]
        manifests = dict(found)

    print(f"Runs merged ({len(run_ids)}):")
    for r in run_ids:
        m = manifests[r]
        print(f"  {r}  levels={m['params']['concurrency_levels']}"
              f"  note={m.get('note')!r}")

    first = manifests[run_ids[0]]
    print(f"\nModel      : {first.get('model')}")
    print(f"Workload   : ISL={first['params']['isl']}, OSL={first['params']['osl']}, "
          f"ignore_eos={first['params']['ignore_eos']}")
    print(f"Commit     : {first.get('git_commit')}")

    rows, dupes = merge(load_summaries(run_ids))

    if dupes:
        print("\nOverlapping levels (averaged):")
        for conc, rids, spread in dupes:
            flag = "" if spread < 5 else "   <-- high spread, check environment"
            print(f"  N={conc:<4} from {len(rids)} runs, "
                  f"throughput spread {spread:.1f}%{flag}")

    # --- Table ---
    print("\n=== Saturation curve ===")
    hdr = (f"{'N':>5}{'sys tok/s':>11}{'user tok/s':>12}{'eff':>7}"
           f"{'req/s':>8}{'TTFT p95':>10}{'p99':>10}{'ITL p95':>9}{'ITL p99':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['concurrency']:>5}{r['sys_out_tok_s']:>11.1f}"
              f"{r['user_speed_tok_s']:>12.1f}{r['efficiency']:>7.3f}"
              f"{r['req_per_s']:>8.2f}{r['ttft_p95']:>10.1f}{r['ttft_p99']:>10.1f}"
              f"{r['itl_p95']:>9.2f}{r['itl_p99']:>9.2f}")

    # --- Headline numbers ---
    base = rows[0]
    peak = max(rows, key=lambda r: r["sys_out_tok_s"])
    print(f"\nSingle-stream baseline : {base['sys_out_tok_s']:.1f} tok/s "
          f"@ N={base['concurrency']}")
    print(f"Peak system throughput : {peak['sys_out_tok_s']:.1f} tok/s "
          f"@ N={peak['concurrency']}  ({peak['sys_out_tok_s']/base['sys_out_tok_s']:.1f}x)")
    print(f"Per-user speed at peak : {peak['user_speed_tok_s']:.1f} tok/s "
          f"({peak['user_speed_tok_s']/base['user_speed_tok_s']*100:.0f}% of single-stream)")

    slo = classify(rows, args.slo_ttft, args.slo_itl)
    print(f"\nSLO targets: TTFT p95 <= {args.slo_ttft:.0f} ms, "
          f"ITL p95 <= {args.slo_itl:.0f} ms")
    if slo:
        print(f"Sustainable capacity   : N={slo['concurrency']}, "
              f"{slo['sys_out_tok_s']:.0f} tok/s, {slo['req_per_s']:.2f} req/s")
        print(f"  headroom vs peak     : {slo['sys_out_tok_s']/peak['sys_out_tok_s']*100:.0f}% "
              f"of peak throughput at {slo['user_speed_tok_s']:.0f} tok/s per user")
    else:
        print("Sustainable capacity   : none of the measured levels meet the SLO")

    # --- CSV ---
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PROC_DIR / f"{args.label}_saturation.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV  : {out_csv}")

    if not args.plot:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: uv pip install matplotlib")
        return

    xs = [r["concurrency"] for r in rows]
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(13, 5))

    # -- Panel 1: throughput vs per-user speed --
    c_sys, c_user = "#1f77b4", "#d62728"
    ax1.plot(xs, [r["sys_out_tok_s"] for r in rows], "o-",
             color=c_sys, lw=2, label="System output throughput")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(x) for x in xs])
    ax1.set_xlabel("Concurrency (in-flight requests)")
    ax1.set_ylabel("System throughput (tokens/s)", color=c_sys)
    ax1.tick_params(axis="y", labelcolor=c_sys)
    ax1.grid(alpha=.3)

    ax2 = ax1.twinx()
    ax2.plot(xs, [r["user_speed_tok_s"] for r in rows], "s--",
             color=c_user, lw=2, label="Per-user speed")
    ax2.set_ylabel("Per-user speed (tokens/s)", color=c_user)
    ax2.tick_params(axis="y", labelcolor=c_user)
    ax2.set_ylim(0, max(r["user_speed_tok_s"] for r in rows) * 1.15)

    if slo:
        ax1.axvline(slo["concurrency"], color="green", ls=":", lw=1.5)
        ax1.annotate(f"SLO limit\nN={slo['concurrency']}",
                     xy=(slo["concurrency"], slo["sys_out_tok_s"]),
                     xytext=(6, -34), textcoords="offset points",
                     fontsize=8, color="green")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax1.set_title("Throughput vs per-user speed", fontsize=11)

    # -- Panel 2: latency, log scale, SLO budgets --
    ax3.plot(xs, [r["ttft_p95"] for r in rows], "o-", color="#ff7f0e",
             lw=2, label="TTFT p95")
    ax3.plot(xs, [r["itl_p95"] for r in rows], "s-", color="#2ca02c",
             lw=2, label="ITL p95")
    ax3.plot(xs, [r["itl_p50"] for r in rows], "^:", color="#2ca02c",
             lw=1.2, alpha=.7, label="ITL p50")
    ax3.axhline(args.slo_ttft, color="#ff7f0e", ls="--", lw=1, alpha=.6)
    ax3.axhline(args.slo_itl, color="#2ca02c", ls="--", lw=1, alpha=.6)
    ax3.text(xs[0], args.slo_ttft * 1.1, f"TTFT budget {args.slo_ttft:.0f} ms",
             fontsize=7.5, color="#ff7f0e")
    ax3.text(xs[0], args.slo_itl * 1.1, f"ITL budget {args.slo_itl:.0f} ms",
             fontsize=7.5, color="#2ca02c")
    ax3.set_xscale("log", base=2)
    ax3.set_yscale("log")
    ax3.set_xticks(xs)
    ax3.set_xticklabels([str(x) for x in xs])
    ax3.set_xlabel("Concurrency (in-flight requests)")
    ax3.set_ylabel("Latency (ms, log scale)")
    ax3.grid(alpha=.3, which="both")
    ax3.legend(fontsize=9)
    ax3.set_title("Latency under load", fontsize=11)

    model = (first.get("model") or "").split("/")[-1]
    fig.suptitle(
        f"T2 — Concurrency saturation — {model} on RTX 4070 Ti SUPER "
        f"(ISL={first['params']['isl']}, OSL={first['params']['osl']})",
        fontsize=11)
    fig.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PLOT_DIR / f"{args.label}_saturation.png"
    fig.savefig(out_png, dpi=150)
    print(f"Plot : {out_png}")


if __name__ == "__main__":
    main()
