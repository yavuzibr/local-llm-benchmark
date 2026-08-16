#!/usr/bin/env python3
"""
T2 — Esszamanlilik doyma egrisi.

Worker havuzu yaklasimi: N paralel dongu, her biri istegi bitirince hemen
yenisini baslatir. Boylece sunucuda surekli N istek ucusta kalir.
"Baslat, hepsini bekle, tekrarla" yaklasiminda esszamanlilik kosunun
sonuna dogru dusuyor ve olculen deger hedeflenenden az oluyor.

Olculen metrikler:
  - Sistem cikti verimi  : toplam cikti token / duvar saati suresi
  - Kullanici basina hiz : 1 / TPOT
  - Verimlilik orani     : sistem verimi / (N x tek kullanici hizi)

Kullanim:
    python scripts/step4_concurrency.py --concurrency 1,2,4,8,16,32,64
    python scripts/step4_concurrency.py --concurrency 1,4,16 --n-per-worker 8
"""

import argparse
import asyncio
import json
import random
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI
from transformers import AutoTokenizer

SCHEMA_VERSION = 1
BASE_URL = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"
MODEL = "LiquidAI/LFM2.5-2.6B"

POOL_TEXT = (
    "Caenorhabditis elegans is a free-living transparent nematode about one "
    "millimetre in length that lives in temperate soil environments. It was "
    "introduced as a model organism for research into developmental biology, "
    "neuroscience and genetics. The hermaphrodite adult has exactly 959 somatic "
    "cells and the complete cell lineage has been mapped. Its nervous system "
    "contains 302 neurons whose connectivity was fully reconstructed from serial "
    "section electron micrographs. Because the body is transparent, individual "
    "cells can be observed in the living animal using differential interference "
    "contrast microscopy. The genome was the first of any multicellular organism "
    "to be sequenced completely. Researchers use forward and reverse genetic "
    "screens, RNA interference, and fluorescent reporters to study gene function. "
    "The short generation time of about three days and the ability to freeze "
    "stocks indefinitely make the organism convenient for laboratory work. "
) * 40


class PromptBuilder:
    def __init__(self, model_id):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.pool = self.tok.encode(POOL_TEXT, add_special_tokens=False)
        self.overhead = self._tlen("")

    def _tlen(self, content):
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        return len(self.tok.encode(text, add_special_tokens=False))

    def build(self, target_isl, rng):
        want = max(1, target_isl - self.overhead)
        start = rng.randrange(0, max(1, len(self.pool) - want - 1))
        text = self.tok.decode(self.pool[start:start + want])
        for _ in range(6):
            total = self._tlen(text)
            diff = target_isl - total
            if abs(diff) <= 2:
                break
            if diff > 0:
                text = text + " " + self.tok.decode(self.pool[start:start + diff + 2])
            else:
                cur = self.tok.encode(text, add_special_tokens=False)
                text = self.tok.decode(cur[:max(1, len(cur) + diff)])
        return text


async def one_request(client, prompt, max_tokens):
    t_send = time.perf_counter()
    t_first = None
    token_times = []
    usage = None
    stream = None
    try:
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.0, stream=True,
            stream_options={"include_usage": True},
            extra_body={"ignore_eos": True},
        )
        async for chunk in stream:
            now = time.perf_counter()
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            role = getattr(d, "role", None)
            content = getattr(d, "content", None)
            if content is None or (role is not None and content == ""):
                continue
            if t_first is None:
                t_first = now
            token_times.append(now)
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if stream is not None:
            try:
                await stream.close()
            except Exception:
                pass

    t_end = time.perf_counter()
    if t_first is None:
        return {"status": "error", "error": "no_content_chunk"}

    return {
        "status": "ok", "error": None,
        "t_send": t_send, "t_end": t_end,
        "ttft_ms": round((t_first - t_send) * 1000, 4),
        "e2e_ms": round((t_end - t_send) * 1000, 4),
        "itls_ms": [round((token_times[i] - token_times[i - 1]) * 1000, 4)
                    for i in range(1, len(token_times))],
        "chunk_count": len(token_times),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
    }


async def run_level(client, pb, rng, conc, isl, osl, n_per_worker, warmup):
    """Belirli bir esszamanlilik seviyesini kosar."""
    # Warm-up: ayni esszamanlilikta, kaydedilmez.
    async def warm():
        for _ in range(warmup):
            await one_request(client, pb.build(isl, rng), osl)
    await asyncio.gather(*[warm() for _ in range(conc)])

    results = []

    async def worker(wid):
        for i in range(n_per_worker):
            rec = await one_request(client, pb.build(isl, rng), osl)
            rec["worker"] = wid
            rec["seq"] = i
            results.append(rec)

    t0 = time.perf_counter()
    await asyncio.gather(*[worker(w) for w in range(conc)])
    t1 = time.perf_counter()

    return results, t1 - t0


def summarize(results, wall_s, conc):
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        return None

    ttfts = [r["ttft_ms"] for r in ok]
    e2es = [r["e2e_ms"] for r in ok]
    tpots = [statistics.mean(r["itls_ms"]) for r in ok if r["itls_ms"]]
    pooled = [x for r in ok for x in r["itls_ms"]]
    out_tokens = sum(r["completion_tokens"] or 0 for r in ok)
    in_tokens = sum(r["prompt_tokens"] or 0 for r in ok)

    def pct(v, p):
        s = sorted(v)
        return s[max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))]

    tpot_mean = statistics.mean(tpots)
    user_speed = 1000 / tpot_mean
    sys_throughput = out_tokens / wall_s

    return {
        "concurrency": conc,
        "n_requests": len(ok),
        "n_errors": len(results) - len(ok),
        "wall_s": wall_s,
        "out_tokens": out_tokens,
        "in_tokens": in_tokens,
        "sys_out_tok_s": sys_throughput,
        "sys_total_tok_s": (in_tokens + out_tokens) / wall_s,
        "req_per_s": len(ok) / wall_s,
        "user_speed_tok_s": user_speed,
        "efficiency": sys_throughput / (conc * user_speed),
        "ttft_p50": pct(ttfts, 50), "ttft_p95": pct(ttfts, 95), "ttft_p99": pct(ttfts, 99),
        "e2e_p50": pct(e2es, 50), "e2e_p95": pct(e2es, 95),
        "tpot_mean": tpot_mean,
        "itl_p50": pct(pooled, 50), "itl_p95": pct(pooled, 95), "itl_p99": pct(pooled, 99),
    }


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def gpu_snapshot():
    q = "name,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm,utilization.gpu"
    out = sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader,nounits")
    if not out:
        return None
    keys = ["name", "memory_total_mib", "memory_used_mib", "temperature_c",
            "power_draw_w", "clock_sm_mhz", "utilization_pct"]
    return dict(zip(keys, [v.strip() for v in out.split(",")]))


def package_versions():
    import importlib.metadata as m
    out = {}
    for p in ["vllm", "torch", "triton", "transformers", "openai"]:
        try:
            out[p] = m.version(p)
        except Exception:
            out[p] = None
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    ap.add_argument("--isl", type=int, default=1024)
    ap.add_argument("--osl", type=int, default=256)
    ap.add_argument("--n-per-worker", type=int, default=5,
                    help="her worker'in yapacagi istek sayisi (>=4 onerilir)")
    ap.add_argument("--warmup", type=int, default=2, help="worker basina warm-up")
    ap.add_argument("--cooldown", type=float, default=15.0, help="seviyeler arasi bekleme (s)")
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--server-cmd", default=None)
    ap.add_argument("--note", default=None)
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    outdir = Path(args.out) / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"run_id : {run_id}")
    print(f"cikti  : {outdir}")
    print("tokenizer yukleniyor...", end="", flush=True)
    pb = PromptBuilder(MODEL)
    rng = random.Random(args.seed)
    print(f" bitti (template ek yuku: {pb.overhead} token)")

    manifest = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "scenario": "T2",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL, "base_url": BASE_URL,
        "git_commit": sh("git rev-parse --short HEAD"),
        "git_dirty": bool(sh("git status --porcelain")),
        "python": sys.version.split()[0], "packages": package_versions(),
        "gpu_idle_before": gpu_snapshot(), "gpu_at_finish": None,
        "params": {
            "concurrency_levels": levels, "isl": args.isl, "osl": args.osl,
            "n_per_worker": args.n_per_worker, "warmup_per_worker": args.warmup,
            "cooldown_s": args.cooldown, "temperature": 0.0, "ignore_eos": True,
        },
        "chat_template_overhead_tokens": pb.overhead,
        "server_cmd": args.server_cmd, "note": args.note,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    jsonl = (outdir / "requests.jsonl").open("w")
    summaries = []

    async with AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY,
                           max_retries=0, timeout=600.0) as client:
        for conc in levels:
            total = conc * args.n_per_worker
            print(f"\nesszamanlilik {conc:>3}  ({total} istek)...", end="", flush=True)
            results, wall = await run_level(
                client, pb, rng, conc, args.isl, args.osl,
                args.n_per_worker, args.warmup)

            for i, r in enumerate(results):
                r.update({
                    "schema_version": SCHEMA_VERSION, "run_id": run_id,
                    "request_id": f"{run_id}-c{conc:03d}-{i:04d}",
                    "scenario": "T2", "concurrency": conc,
                    "isl_target": args.isl, "osl_target": args.osl,
                })
                r.pop("t_send", None)
                r.pop("t_end", None)
                jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")
            jsonl.flush()

            s = summarize(results, wall, conc)
            if s:
                s["gpu"] = gpu_snapshot()
                summaries.append(s)
                print(f" {wall:5.1f}s | sistem {s['sys_out_tok_s']:7.1f} tok/s"
                      f" | kullanici {s['user_speed_tok_s']:6.1f} tok/s"
                      f" | TTFT p95 {s['ttft_p95']:7.1f} ms")
            if conc != levels[-1] and args.cooldown:
                await asyncio.sleep(args.cooldown)

    jsonl.close()
    manifest["gpu_at_finish"] = gpu_snapshot()
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (outdir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))

    # --- Tablo ---
    print("\n=== T2 doyma egrisi ===")
    hdr = (f"{'N':>4}{'sistem':>10}{'kullanici':>11}{'verimlilik':>12}"
           f"{'istek/s':>9}{'TTFT p50':>10}{'p95':>9}{'p99':>9}{'ITL p95':>9}")
    print(hdr)
    print("-" * len(hdr))
    base = summaries[0]["user_speed_tok_s"] if summaries else 1
    for s in summaries:
        print(f"{s['concurrency']:>4}{s['sys_out_tok_s']:>10.1f}"
              f"{s['user_speed_tok_s']:>11.1f}{s['efficiency']:>12.3f}"
              f"{s['req_per_s']:>9.2f}{s['ttft_p50']:>10.1f}"
              f"{s['ttft_p95']:>9.1f}{s['ttft_p99']:>9.1f}{s['itl_p95']:>9.2f}"
              + ("" if s["n_errors"] == 0 else f"  [{s['n_errors']} hata]"))

    if len(summaries) > 1:
        best = max(summaries, key=lambda s: s["sys_out_tok_s"])
        print(f"\nen yuksek sistem verimi : {best['sys_out_tok_s']:.1f} tok/s "
              f"(N={best['concurrency']})")
        print(f"tek akisa gore kazanc   : {best['sys_out_tok_s']/summaries[0]['sys_out_tok_s']:.1f}x")
        print(f"o noktada kullanici hizi: {best['user_speed_tok_s']:.1f} tok/s "
              f"({best['user_speed_tok_s']/base*100:.0f}% of tek akis)")

    print(f"\nham veri : {outdir/'requests.jsonl'}")
    print(f"ozet     : {outdir/'summary.json'}")


if __name__ == "__main__":
    asyncio.run(main())
