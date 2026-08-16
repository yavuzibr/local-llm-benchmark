# local-llm-benchmark

Yerel donanımda vLLM ile çalıştırılan LLM'lerin performans ölçümleri.

**Donanım:** RTX 4070 Ti SUPER (16 GB) · Windows + WSL2 · Ubuntu 26.04
**Motor:** vLLM 0.26.0 · PyTorch 2.11.0+cu130 · Python 3.12.14

## Durum

| Test | Açıklama | Durum |
|---|---|---|
| T0 | Kapasite envanteri | ✅ |
| T1 | Tek akış gecikmesi (ISL × OSL) | ✅ |
| T2 | Eşzamanlılık doyma eğrisi | 🚧 |
| T3 | Context ölçekleme | ⬜ |
| Capability | Model yeteneği testleri | ⬜ |

## Modeller

- [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) — BF16, hibrit mimari (22 conv + 8 GQA katman)

## İlk bulgular (LFM2.5-2.6B)

| Metrik | Değer |
|---|---|
| Decode hızı (tek akış) | 107 token/s |
| Bellek bant genişliği tavanının kullanımı | %86 |
| Prefill hızı | 14.673 token/s |
| KV cache kapasitesi | 488.404 token |
| Context kaybı (128 → 8k) | %2.6 |

## Yapı
## Kullanım

```bash
# Sunucu
vllm serve LiquidAI/LFM2.5-2.6B --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 32768 \
  --kv-cache-memory=6442450944 --no-enable-prefix-caching --seed 0

# Ölçüm
python scripts/step3_grid.py --scenario T1 --isl 128,1024,4096 --osl 128,512 --n 10

# Analiz
python scripts/analyze.py --plot
```

Ölçüm yöntemi ve her kararın gerekçesi: [docs/METHODOLOGY.md](docs/METHODOLOGY.md)

