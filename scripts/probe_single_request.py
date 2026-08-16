#!/usr/bin/env python3
"""
Adım 1 — Tek istek, elle ölçüm.

Amaç: TTFT ve ITL'i client tarafında, streaming üzerinden ölçmek.
Bu script kasıtlı olarak basit tutuldu: dosyaya yazmıyor, ortalama almıyor,
tek bir isteği ölçüp ekrana döküyor. Önce sayıları gözle görüp mantıklı
olduklarına ikna olacağız, sonra Adım 2'de tekrarlamaya geçeceğiz.

Kullanım:
    python scripts/probe_single_request.py
    python scripts/probe_single_request.py --max-tokens 1024
    python scripts/probe_single_request.py --isl 1024 --max-tokens 256 --ignore-eos
"""

import argparse
import asyncio
import statistics
import time

from openai import AsyncOpenAI

BASE_URL = "http://127.0.0.1:8000/v1"
# openai >= 2.34 boş string'i reddediyor ("Missing credentials"),
# yerel sunucu kimlik doğrulaması istemese de dolu bir değer şart.
API_KEY = "EMPTY"
MODEL = "LiquidAI/LFM2.5-2.6B"


def make_prompt(target_tokens: int) -> str:
    """
    Kaba bir ISL kontrolü. Gerçek token sayısı tokenizer'a bağlı, o yüzden
    hedef sayıyı garanti etmiyoruz -- sunucunun raporladığı prompt_tokens'a
    bakacağız. Adım 3'te bunu tokenizer ile hassaslaştıracağız.

    Benzersiz bir önek ekliyoruz ki prefix cache'e denk gelmesin
    (sunucuda kapalı olsa bile alışkanlık olarak).
    """
    uniq = f"[run-{time.time_ns()}] "
    if target_tokens <= 0:
        return uniq + "C. elegans nedir? Kısaca açıkla."
    filler = "kelime " * max(1, int(target_tokens * 0.9))
    return uniq + "Aşağıdaki metni özetle:\n" + filler


async def measure(client, prompt, max_tokens, ignore_eos):
    extra_body = {}
    if ignore_eos:
        # Modelin erken durmasını engeller: çıktı uzunluğunu biz sabitleriz.
        extra_body["ignore_eos"] = True

    t_send = time.perf_counter()
    t_first = None
    token_times = []          # içerik taşıyan her chunk'ın varış anı
    text_parts = []
    usage = None
    empty_leading_chunks = 0

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
        # Son chunk'ta sunucunun kendi token sayacını almamızı sağlar.
        # Client tarafında tokenizer ile saymak hataya açık -- buna güveniyoruz.
        stream_options={"include_usage": True},
        extra_body=extra_body or None,
    )

    async for chunk in stream:
        now = time.perf_counter()

        if getattr(chunk, "usage", None):
            usage = chunk.usage

        if not chunk.choices:
            # include_usage acikken son chunk'ta choices bos gelir.
            continue

        delta = chunk.choices[0].delta
        role = getattr(delta, "role", None)
        content = getattr(delta, "content", None)

        # Acilis chunk'i: role="assistant", content="". Token tasimaz.
        # Bunu TTFT'ye saymak yaygin bir olcum hatasidir.
        if content is None or (role is not None and content == ""):
            if t_first is None:
                empty_leading_chunks += 1
            continue

        if t_first is None:
            t_first = now
        token_times.append(now)
        text_parts.append(content)

    t_end = time.perf_counter()

    if t_first is None:
        raise RuntimeError("Hiç içerik chunk'ı gelmedi -- sunucu yanıtını kontrol et.")

    # ITL: ardışık içerik chunk'ları arasındaki süreler.
    itls_ms = [
        (token_times[i] - token_times[i - 1]) * 1000
        for i in range(1, len(token_times))
    ]

    return {
        "ttft_ms": (t_first - t_send) * 1000,
        "e2e_ms": (t_end - t_send) * 1000,
        "itls_ms": itls_ms,
        "chunk_count": len(token_times),
        "empty_leading_chunks": empty_leading_chunks,
        "usage": usage,
        "text": "".join(text_parts),
    }


def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def report(r, show_text):
    u = r["usage"]
    prompt_tokens = getattr(u, "prompt_tokens", None) if u else None
    completion_tokens = getattr(u, "completion_tokens", None) if u else None

    itls = r["itls_ms"]
    print("\n=== Ölçüm ===")
    print(f"TTFT                : {r['ttft_ms']:8.1f} ms")
    print(f"E2E                 : {r['e2e_ms']:8.1f} ms")
    if itls:
        print(f"ITL  p50 / p95 / max: {pct(itls,50):6.1f} / {pct(itls,95):6.1f} / {max(itls):6.1f} ms")
        print(f"ITL  ortalama(TPOT) : {statistics.mean(itls):8.2f} ms")
        print(f"Kullanıcı hızı      : {1000/statistics.mean(itls):8.1f} token/s")

    print("\n=== Token sayıları ===")
    print(f"prompt_tokens (sunucu)     : {prompt_tokens}")
    print(f"completion_tokens (sunucu) : {completion_tokens}")
    print(f"içerik chunk sayısı        : {r['chunk_count']}")
    print(f"atlanan boş baş chunk      : {r['empty_leading_chunks']}")

    if completion_tokens is not None and completion_tokens != r["chunk_count"]:
        print(
            "\n[!] chunk sayısı ile completion_tokens eşleşmiyor.\n"
            "    Bir chunk birden fazla token taşıyor olabilir. Bu durumda ITL,\n"
            "    'token başına süre' değil 'chunk başına süre' demektir --\n"
            "    metodoloji dosyasına bu notu düşmemiz gerekir."
        )

    if show_text:
        print("\n=== Çıktı ===")
        print(r["text"])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isl", type=int, default=0, help="hedef prompt uzunluğu (kaba)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--ignore-eos", action="store_true",
                    help="modelin erken durmasını engelle (çıktı uzunluğunu sabitler)")
    ap.add_argument("--show-text", action="store_true", help="modelin çıktısını da yazdır")
    args = ap.parse_args()

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    try:
        r = await measure(client, make_prompt(args.isl), args.max_tokens, args.ignore_eos)
        report(r, args.show_text)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
