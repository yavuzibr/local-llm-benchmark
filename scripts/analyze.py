#!/usr/bin/env python3
"""
Adım 4 — Analiz: ham JSONL -> persentil tablosu + islenmis CSV.

Ham veriyi hic degistirmez. Tum ozetler bu scriptle yeniden uretilebilir.

Kullanim:
    # En son kosuyu analiz et
    python scripts/analyze.py

    # Belirli bir kosuyu
    python scripts/analyze.py --run 20260815T183104Z-ee97f8

    # Grafik de uret (matplotlib gerekir: uv pip install matplotlib)
    python scripts/analyze.py --plot
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

RAW_DIR = Path("results/raw")
PROC_DIR = Path("results/processed")
PLOT_DIR = Path("results/plots")


def pct(values, p):
    """Nearest-rank persentil."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = int(round((p / 100) * (len(s) - 1)))
    return s[max(0, min(len(s) - 1, k))]


def load(run_id):
    path = RAW_DIR / run_id / "requests.jsonl"
    records = []
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "ok":
                records.append(r)
    manifest = json.loads((RAW_DIR / run_id / "manifest.json").read_text())
    return records, manifest


def cell_stats(records):
    """Bir hucrenin (ISL x OSL) istatistiklerini hesaplar."""
    ttfts = [r["ttft_ms"] for r in records]
    e2es = [r["e2e_ms"] for r in records]
    tpots = [statistics.mean(r["itls_ms"]) for r in records if r["itls_ms"]]
    # Token seviyesi: TUM isteklerin ITL'leri tek havuzda toplanir.
    # Her istegin persentilini alip ortalamak matematiksel olarak yanlistir.
    pooled = [x for r in records for x in r["itls_ms"]]

    isl = statistics.median([r["prompt_tokens"] for r in records])
    osl = statistics.median([r["completion_tokens"] for r in records])

    tpot_mean = statistics.mean(tpots)
    return {
        "isl_target": records[0]["isl_target"],
        "osl_target": records[0]["osl_target"],
        "isl_actual": isl,
        "osl_actual": osl,
        "n": len(records),
        "ttft_p50": pct(ttfts, 50),
        "ttft_p95": pct(ttfts, 95),
        "ttft_p99": pct(ttfts, 99),
        "e2e_p50": pct(e2es, 50),
        "e2e_p95": pct(e2es, 95),
        "tpot_mean": tpot_mean,
        "tpot_cv_pct": (statistics.stdev(tpots) / tpot_mean * 100) if len(tpots) > 1 else 0.0,
        "itl_p50": pct(pooled, 50),
        "itl_p95": pct(pooled, 95),
        "itl_p99": pct(pooled, 99),
        "itl_max": max(pooled) if pooled else float("nan"),
        "speed_tok_s": 1000 / tpot_mean,
        "n_itl": len(pooled),
    }


def linfit(xs, ys):
    """En kucuk kareler: y = a + b*x. numpy olmadan."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else float("nan")
    a = my - b * mx
    # R^2
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return a, b, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run_id (varsayilan: en son kosu)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    run_id = args.run
    if run_id is None:
        runs = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())
        if not runs:
            raise SystemExit("results/raw altinda kosu bulunamadi")
        run_id = runs[-1]

    records, manifest = load(run_id)
    print(f"run_id   : {run_id}")
    print(f"senaryo  : {manifest.get('scenario')}")
    print(f"commit   : {manifest.get('git_commit')}")
    print(f"istek    : {len(records)} basarili")

    gb = manifest.get("gpu_before") or {}
    ga = manifest.get("gpu_after") or {}
    if gb and ga:
        print(f"GPU      : {gb.get('temperature_c')}C -> {ga.get('temperature_c')}C, "
              f"{gb.get('power_draw_w')}W -> {ga.get('power_draw_w')}W")

    # Hucrelere ayir
    cells = {}
    for r in records:
        cells.setdefault((r["isl_target"], r["osl_target"]), []).append(r)
    rows = [cell_stats(v) for k, v in sorted(cells.items())]

    # --- Tablo ---
    print("\n=== Hucre ozeti ===")
    hdr = (f"{'ISL':>6}{'OSL':>6}{'n':>4}"
           f"{'TTFT p50':>10}{'p95':>9}{'p99':>9}"
           f"{'TPOT':>8}{'tok/s':>8}"
           f"{'ITL p99':>9}{'CV%':>7}")
    print(hdr)
    print("-" * len(hdr))
    for c in rows:
        print(f"{c['isl_actual']:>6.0f}{c['osl_actual']:>6.0f}{c['n']:>4}"
              f"{c['ttft_p50']:>10.1f}{c['ttft_p95']:>9.1f}{c['ttft_p99']:>9.1f}"
              f"{c['tpot_mean']:>8.2f}{c['speed_tok_s']:>8.1f}"
              f"{c['itl_p99']:>9.2f}{c['tpot_cv_pct']:>7.2f}")

    # --- Prefill regresyonu ---
    # TTFT = a + b*ISL. a = sabit ek yuk (HTTP + scheduler + harness),
    # 1/b = saf prefill hizi (token/s).
    xs = [c["isl_actual"] for c in rows]
    ys = [c["ttft_p50"] for c in rows]
    if len(set(xs)) >= 3:
        a, b, r2 = linfit(xs, ys)
        print("\n=== Prefill olceklemesi (TTFT = a + b x ISL) ===")
        print(f"sabit ek yuk (a)   : {a:8.2f} ms")
        print(f"token basina (b)   : {b*1000:8.4f} us/token")
        print(f"prefill hizi (1/b) : {1000/b:8.0f} token/s")
        print(f"R^2                : {r2:8.4f}"
              + ("  (dogrusal)" if r2 > 0.99 else "  <-- dogrusaldan sapma var"))

    # --- Context'in decode'a etkisi ---
    by_isl = {}
    for c in rows:
        by_isl.setdefault(c["isl_actual"], []).append(c["tpot_mean"])
    if len(by_isl) >= 3:
        pts = sorted((k, statistics.mean(v)) for k, v in by_isl.items())
        base = pts[0][1]
        print("\n=== Context'in decode hizina etkisi ===")
        print(f"{'ISL':>8}{'TPOT ms':>10}{'tok/s':>9}{'kayip':>9}")
        for isl, tp in pts:
            print(f"{isl:>8.0f}{tp:>10.3f}{1000/tp:>9.1f}{(tp/base-1)*100:>8.1f}%")

    # --- CSV ---
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PROC_DIR / f"{run_id}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV: {out_csv}")

    # --- Grafik ---
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib yok. Kurmak icin: uv pip install matplotlib")
            return

        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        osls = sorted({c["osl_target"] for c in rows})

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for osl in osls:
            sel = [c for c in rows if c["osl_target"] == osl]
            sel.sort(key=lambda c: c["isl_actual"])
            ax[0].plot([c["isl_actual"] for c in sel],
                       [c["ttft_p50"] for c in sel], "o-", label=f"OSL={osl}")
            ax[1].plot([c["isl_actual"] for c in sel],
                       [c["speed_tok_s"] for c in sel], "o-", label=f"OSL={osl}")
        ax[0].set(xlabel="ISL (token)", ylabel="TTFT p50 (ms)", title="Prefill olceklemesi")
        ax[1].set(xlabel="ISL (token)", ylabel="token/s", title="Decode hizi")
        for a_ in ax:
            a_.grid(alpha=.3)
            a_.legend()
        fig.suptitle(f"T1 — {manifest.get('model')} — {run_id}", fontsize=9)
        fig.tight_layout()
        out_png = PLOT_DIR / f"{run_id}_t1.png"
        fig.savefig(out_png, dpi=140)
        print(f"Grafik: {out_png}")


if __name__ == "__main__":
    main()
