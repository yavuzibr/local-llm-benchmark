#!/usr/bin/env python3
"""
Adım 2 — Tekrarla ve dağılıma bak.

Adım 1'den farkı:
  - Tek bir kalıcı AsyncOpenAI istemcisi kullanır (client ek yükü ~170 ms
    yerine ~0). Bkz. METHODOLOGY 4.2.
  - Warm-up istekleri gönderir ve bunları HİÇBİR istatistiğe katmaz.
  - N istek koşup persentil dağılımı çıkarır.
  - İki ayrı toplama seviyesi raporlar: istek seviyesi ve token seviyesi.

Henüz dosyaya yazmıyor -- ham JSONL çıktısı Adım 3'te gelecek.

Kullanım:
    python scripts/step2_repeat.py
    python scripts/step2_repeat.py --n 30 --max-tokens 256
    python scripts/step2_repeat.py --isl 1024 --n 20
"""

import argparse
import asyncio
import statistics
import time

from openai import AsyncOpenAI

BASE_URL = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"  # openai >= 2.34 bos string kabul etmiyor
MODEL = "LiquidAI/LFM2.5-2.6B"


def make_prompt(target_tokens: int, tag: str) -> str:
    """
    Her istege benzersiz bir onek eklenir: prefix cache'e denk gelmemesi
    icin. Sunucuda prefix caching kapali olsa da bu aliskanligi koruyoruz.
    """
    uniq = f"[{tag}-{time.time_ns()}] "
    if target_tokens <= 0:
        return uniq + "C. elegans nedir? Kisaca acikla."
    filler = "kelime " * max(1, int(target_tokens * 0.9))
    return uniq + "Asagidaki metni ozetle:\n" + filler


async def one_request(client, prompt, max_tokens):
    t_send = time.perf_counter()
    t_first = None
    token_times = []
    usage = None

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

    t_end = time.perf_counter()

    if t_first is None:
        raise RuntimeError("Hic icerik chunk'i gelmedi.")

    itls_ms = [
        (token_times[i] - token_times[i - 1]) * 1000
        for i in range(1, len(token_times))
    ]

    return {
        "ttft_ms": (t_first - t_send) * 1000,
        "e2e_ms": (t_end - t_send) * 1000,
        "itls_ms": itls_ms,
        "chunk_count": len(token_times),
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
    }


def pct(values, p):
    """Nearest-rank persentil. Bos listede nan doner."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = int(round((p / 100) * (len(s) - 1)))
    return s[max(0, min(len(s) - 1, k))]


def row(label, values, fmt="{:9.1f}"):
    cells = [pct(values, p) for p in (50, 90, 95, 99)]
    cells += [min(values), max(values)]
    print(f"{label:<16}" + "".join(fmt.format(c) for c in cells))


def report(records, args):
    n = len(records)

    ttfts = [r["ttft_ms"] for r in records]
    e2es = [r["e2e_ms"] for r in records]
    tpots = [statistics.mean(r["itls_ms"]) for r in records if r["itls_ms"]]
    speeds = [1000 / t for t in tpots]

    # Token seviyesi: TUM isteklerin ITL'leri tek havuzda.
    # Her istegin persentilini alip ortalamak matematiksel olarak yanlistir.
    pooled_itls = [x for r in records for x in r["itls_ms"]]

    print("\n=== Konfigurasyon ===")
    print(f"warm-up  : {args.warmup} (istatistiklere dahil DEGIL)")
    print(f"olculen  : {n} istek")
    print(f"ISL hedef: {args.isl or 'kisa'} | max_tokens: {args.max_tokens} | ignore_eos: True")

    print(f"\n=== Istek seviyesi (n={n}) ===")
    print(f"{'':<16}{'p50':>9}{'p90':>9}{'p95':>9}{'p99':>9}{'min':>9}{'max':>9}")
    row("TTFT (ms)", ttfts)
    row("E2E (ms)", e2es)
    row("TPOT (ms)", tpots, "{:9.2f}")
    row("Hiz (tok/s)", speeds)

    print(f"\n=== Token seviyesi ITL (n={len(pooled_itls)} aralik) ===")
    print(f"{'':<16}{'p50':>9}{'p90':>9}{'p95':>9}{'p99':>9}{'min':>9}{'max':>9}")
    row("ITL (ms)", pooled_itls, "{:9.2f}")

    print("\n=== Tutarlilik kontrolleri ===")
    pt = {r["prompt_tokens"] for r in records}
    ct = {r["completion_tokens"] for r in records}
    mismatch = sum(1 for r in records if r["chunk_count"] != r["completion_tokens"])
    print(f"prompt_tokens degerleri     : {sorted(pt)}")
    print(f"completion_tokens degerleri : {sorted(ct)}")
    print(f"chunk == completion_tokens  : {n - mismatch}/{n}" + ("  OK" if mismatch == 0 else "  <-- SORUN"))

    # Kararlilik: TPOT'un varyasyon katsayisi.
    if len(tpots) > 1:
        cv = statistics.stdev(tpots) / statistics.mean(tpots) * 100
        print(f"TPOT varyasyon katsayisi    : {cv:.2f}%" +
              ("  (kararli)" if cv < 5 else "  <-- yuksek, ortami kontrol et"))

    # TTFT'de ilk istek hala sicrama yapiyor mu?
    if n > 1 and ttfts[0] > 2 * statistics.median(ttfts[1:]):
        print(f"\n[!] Ilk olculen istegin TTFT'si ({ttfts[0]:.1f} ms) medyanin 2 katindan fazla.")
        print("    Warm-up sayisi yetersiz olabilir -- --warmup degerini artir.")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="olculecek istek sayisi")
    ap.add_argument("--warmup", type=int, default=5, help="warm-up istek sayisi")
    ap.add_argument("--isl", type=int, default=0, help="hedef prompt uzunlugu (kaba)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--sleep", type=float, default=0.0, help="istekler arasi bekleme (s)")
    args = ap.parse_args()

    async with AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY) as client:
        print(f"warm-up: {args.warmup} istek...", end="", flush=True)
        for i in range(args.warmup):
            await one_request(client, make_prompt(args.isl, f"warm{i}"), args.max_tokens)
        print(" bitti")

        print(f"olcum: {args.n} istek...", end="", flush=True)
        records = []
        for i in range(args.n):
            records.append(
                await one_request(client, make_prompt(args.isl, f"m{i}"), args.max_tokens)
            )
            if args.sleep:
                await asyncio.sleep(args.sleep)
        print(" bitti")

    report(records, args)


if __name__ == "__main__":
    asyncio.run(main())
