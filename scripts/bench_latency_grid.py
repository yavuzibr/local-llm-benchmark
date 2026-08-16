#!/usr/bin/env python3
"""
Benchmark: latency across an ISL x OSL grid (tests T1 and T3).

Collects raw data. For each cell it runs warm-up requests, then N measured
requests at concurrency 1, appending one JSON line per request to
results/raw/<run_id>/requests.jsonl.

Input length is controlled with the model's own tokenizer: the chat template
overhead is measured, subtracted from the target, and the remainder is taken
from a random window of a text pool. The random window keeps every prompt
unique, so no request can benefit from a warm prefix.

Each run also writes manifest.json recording the git commit, package versions,
GPU state and every parameter used.

Raw data is never modified. Summaries are produced separately by
analyze_grid.py and can always be regenerated from the JSONL.

Usage:
    # T1 single-stream latency grid
    python scripts/bench_latency_grid.py --scenario T1 --isl 128,1024,4096 --osl 128,512 --n 10

    # Single cell
    python scripts/bench_latency_grid.py --scenario smoke --isl 1024 --osl 128 --n 5
"""

import argparse
import asyncio
import json
import os
import random
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
API_KEY = "EMPTY"  # openai >= 2.34 bos string kabul etmiyor
MODEL = "LiquidAI/LFM2.5-2.6B"

# ISL dolgusu icin metin havuzu. Icerigi onemli degil; onemli olan
# yeterince uzun ve tekrar etmeyen bir token dizisi vermesi.
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
) * 1400


class PromptBuilder:
    """Hedef token sayisini tutturan, her cagrida benzersiz prompt uretir."""

    def __init__(self, model_id: str):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.pool = self.tok.encode(POOL_TEXT, add_special_tokens=False)
        if len(self.pool) < 4096:
            raise RuntimeError("POOL_TEXT cok kisa, uzun ISL testleri icin buyutun.")
        # Chat template'in kendi ek yuku: bos bir kullanici mesajinin maliyeti.
        self.overhead = self._templated_len("")

    def _templated_len(self, user_content: str) -> int:
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(self.tok.encode(text, add_special_tokens=False))

    def build(self, target_isl: int, rng: random.Random) -> tuple[str, int]:
        """
        Hedef ISL'e (template dahil toplam prompt token sayisi) yakin bir
        kullanici mesaji uretir. (metin, tahmini_toplam_token) dondurur.
        Gercek deger sunucudan okunacak; bu sadece hedefe yaklastirmak icin.
        """
        want = max(1, target_isl - self.overhead)
        # Havuzdan rastgele bir pencere: hem benzersiz hem tam uzunlukta.
        start = rng.randrange(0, max(1, len(self.pool) - want - 1))
        ids = self.pool[start:start + want]
        text = self.tok.decode(ids)

        # Decode/encode gidip gelmesi birkac token kaydirabilir. Duzelt.
        for _ in range(6):
            total = self._templated_len(text)
            diff = target_isl - total
            if abs(diff) <= 2:
                break
            if diff > 0:
                extra = self.pool[start:start + diff + 2]
                text = text + " " + self.tok.decode(extra)
            else:
                cur = self.tok.encode(text, add_special_tokens=False)
                text = self.tok.decode(cur[:max(1, len(cur) + diff)])
        return text, self._templated_len(text)


async def one_request(client, prompt, max_tokens):
    """Tek bir streaming istegi olcer. Zaman damgalari perf_counter tabanli."""
    t_send = time.perf_counter()
    t_first = None
    token_times = []
    usage = None
    stream = None

    try:
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"ignore_eos": True},
        )

        async for chunk in stream:
            now = time.perf_counter()

            if getattr(chunk, "usage", None):
                usage = chunk.usage

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            role = getattr(delta, "role", None)
            content = getattr(delta, "content", None)

            # Acilis chunk'i: role="assistant", content="". Token tasimaz.
            if content is None or (role is not None and content == ""):
                continue

            if t_first is None:
                t_first = now
            token_times.append(now)

    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        # httpcore2'nin kapanis uyarisini onlemek icin stream'i acikca kapat.
        if stream is not None:
            try:
                await stream.close()
            except Exception:
                pass

    t_end = time.perf_counter()
    if t_first is None:
        return {"status": "error", "error": "no_content_chunk"}

    itls_ms = [
        round((token_times[i] - token_times[i - 1]) * 1000, 4)
        for i in range(1, len(token_times))
    ]

    return {
        "status": "ok",
        "error": None,
        "ttft_ms": round((t_first - t_send) * 1000, 4),
        "e2e_ms": round((t_end - t_send) * 1000, 4),
        "itls_ms": itls_ms,
        "chunk_count": len(token_times),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
    }


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def gpu_snapshot():
    q = ("name,memory.total,memory.used,temperature.gpu,"
         "power.draw,clocks.sm,utilization.gpu")
    out = sh(f"nvidia-smi --query-gpu={q} --format=csv,noheader,nounits")
    if not out:
        return None
    vals = [v.strip() for v in out.split(",")]
    keys = ["name", "memory_total_mib", "memory_used_mib", "temperature_c",
            "power_draw_w", "clock_sm_mhz", "utilization_pct"]
    return dict(zip(keys, vals))


def package_versions():
    import importlib.metadata as m
    out = {}
    for p in ["vllm", "torch", "triton", "transformers", "openai", "numpy"]:
        try:
            out[p] = m.version(p)
        except Exception:
            out[p] = None
    return out


def build_manifest(args, run_id, isls, osls, prompt_overhead):
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "scenario": args.scenario,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "base_url": BASE_URL,
        "git_commit": sh("git rev-parse --short HEAD"),
        "git_dirty": bool(sh("git status --porcelain")),
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "gpu_before": gpu_snapshot(),
        "gpu_after": None,          # kosu sonunda doldurulur
        "params": {
            "isl_targets": isls,
            "osl_targets": osls,
            "n_per_cell": args.n,
            "warmup": args.warmup,
            "cooldown_s": args.cooldown,
            "temperature": 0.0,
            "ignore_eos": True,
            "concurrency": 1,
        },
        "chat_template_overhead_tokens": prompt_overhead,
        "server_cmd": args.server_cmd,
        "note": args.note,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="T1")
    ap.add_argument("--isl", default="128,1024,4096", help="virgulle ayrilmis hedef ISL")
    ap.add_argument("--osl", default="128,512", help="virgulle ayrilmis hedef OSL")
    ap.add_argument("--n", type=int, default=10, help="hucre basina olculen istek")
    ap.add_argument("--warmup", type=int, default=5, help="hucre basina warm-up")
    ap.add_argument("--cooldown", type=float, default=0.0, help="hucreler arasi bekleme (s)")
    ap.add_argument("--out", default="results/raw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--server-cmd", default=None,
                    help="sunucuyu baslattigin komut (manifest'e kaydedilir)")
    ap.add_argument("--note", default=None)
    args = ap.parse_args()

    isls = [int(x) for x in args.isl.split(",") if x.strip()]
    osls = [int(x) for x in args.osl.split(",") if x.strip()]

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    outdir = Path(args.out) / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"run_id  : {run_id}")
    print(f"cikti   : {outdir}")
    print("tokenizer yukleniyor...", end="", flush=True)
    pb = PromptBuilder(MODEL)
    print(f" bitti (template ek yuku: {pb.overhead} token)")

    manifest = build_manifest(args, run_id, isls, osls, pb.overhead)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    jsonl = (outdir / "requests.jsonl").open("w")
    rng = random.Random(args.seed)
    total_cells = len(isls) * len(osls)
    cell_no = 0
    n_ok = n_err = 0

    async with AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY) as client:
        for isl in isls:
            for osl in osls:
                cell_no += 1
                print(f"\n[{cell_no}/{total_cells}] ISL={isl} OSL={osl}")

                # Warm-up: ayni sekle sahip istekler, hicbiri kaydedilmez.
                print("  warm-up...", end="", flush=True)
                for _ in range(args.warmup):
                    text, _ = pb.build(isl, rng)
                    await one_request(client, text, osl)
                print(" ok")

                print("  olcum  ...", end="", flush=True)
                for i in range(args.n):
                    text, isl_est = pb.build(isl, rng)
                    rec = await one_request(client, text, osl)
                    rec.update({
                        "schema_version": SCHEMA_VERSION,
                        "run_id": run_id,
                        "request_id": f"{run_id}-{cell_no:02d}-{i:04d}",
                        "scenario": args.scenario,
                        "concurrency": 1,
                        "isl_target": isl,
                        "isl_estimated": isl_est,
                        "osl_target": osl,
                        "t_wall_utc": datetime.now(timezone.utc).isoformat(),
                    })
                    jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    jsonl.flush()
                    if rec["status"] == "ok":
                        n_ok += 1
                    else:
                        n_err += 1
                        print(f"\n  [!] hata: {rec['error']}")
                print(" ok")

                # Hizli ozet: hucrenin dogru sekilde kosup kosmadigini gormek icin.
                # Asil analiz Adim 4'te ayri bir scriptle yapilacak.
                if args.cooldown:
                    await asyncio.sleep(args.cooldown)

    jsonl.close()
    manifest["gpu_after"] = gpu_snapshot()
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["requests_ok"] = n_ok
    manifest["requests_error"] = n_err
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\nbitti: {n_ok} basarili, {n_err} hatali")
    print(f"ham veri : {outdir / 'requests.jsonl'}")
    print(f"manifest : {outdir / 'manifest.json'}")


if __name__ == "__main__":
    asyncio.run(main())
