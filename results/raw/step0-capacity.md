# T0 — Kapasite envanteri

**Model:** LiquidAI/LFM2.5-2.6B (BF16)
**Tarih:** 2026-08-15
**Donanım:** RTX 4070 Ti SUPER, 16.376 MiB
**vLLM:** 0.26.0

Serve komutu:

```bash
vllm serve LiquidAI/LFM2.5-2.6B \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching --seed 0
```

> Not: Sonraki koşularda `--gpu-memory-utilization 0.85` yerine
> `--kv-cache-memory=6442450944` kullanılacaktır (gerekçe: METHODOLOGY 5.2).

---

## 1. Bellek yerleşimi

Sunucu açılışında kart üzerinde 14,73 / 15,99 GiB boştu.
Hedeflenen kullanım 0.85 × 15,99 = 13,59 GiB.

| Kalem | GiB |
|---|---|
| Model ağırlıkları | 5,05 |
| Tepe aktivasyon | 0,50 |
| Torch dışı bellek | 0,07 |
| CUDA graph | 0,33 |
| **KV cache** | **7,46** |
| **Toplam** | **13,41** |

Sunucu ayaktayken `nvidia-smi`: 15.050 MiB kullanımda
(vLLM 13,41 GiB + Windows masaüstü ~1 GiB). Kartta ~1.326 MiB boşluk
kaldı — bu, ölçüm sırasında masaüstünde uygulama açılırsa OOM riski
demektir. Bu nedenle sonraki koşularda KV cache 6 GiB'a sabitlenecek.

vLLM'in kendi önerisi loglardan:
`--kv-cache-memory=9264693248` (8,63 GiB) ile bellek tam olarak
doldurulabilirdi; tampon bırakmak için tercih edilmedi.

---

## 2. KV cache kapasitesi

```
GPU KV cache size: 488.404 tokens
Maximum concurrency for 32.768 tokens per request: 14,90x
```

Token başına KV maliyeti ≈ 16 KB.

Bu, aynı boyutta tam-attention bir modelin yaklaşık dörtte biridir.
Sebep mimaridir: LFM2.5-2.6B'nin 30 katmanının 22'si short convolution
bloğu, yalnızca 8'i GQA attention. KV cache sadece attention
katmanlarından doğar.

Kapasite tablosu (7,46 GiB KV cache ile):

| İstek başına context | Eşzamanlı istek |
|---|---|
| 1k | ~380 |
| 8k | ~60 |
| 32k | 14,9 (logda doğrulanmış) |
| 128k | ~3,7 |

**Test tasarımına etkisi:**

- T2 (eşzamanlılık) testinde ISL=1024 ile 64 eşzamanlılığa çıkıldığında
  KV cache darboğaz olmayacaktır. Doyma noktasını `--max-num-seqs` ve
  saf hesap gücü belirleyecektir. `--max-num-seqs` açıkça sabitlenmelidir.
- T3 (context) testi 32k'da kesilmeyip 64k ve 128k'ya kadar
  götürülebilir.

Uyarı: loglar `Add 2 padding layers, may waste at most 9.09% KV cache
memory` diyor. Hibrit mimarinin katman hizalaması nedeniyle KV cache'in
bir kısmı kullanılamıyor.

---

## 3. Cold start (soğuk derleme)

| Aşama | Süre |
|---|---|
| Ağırlık yükleme | 2,02 s |
| Model yükleme (toplam) | 3,52 s |
| torch.compile | 9,00 s |
| CUDA graph yakalama | 4 s (0,33 GiB) |
| Bellek profilleme + dummy run | ~50 s |
| **init engine (toplam)** | **66,09 s** |

Bu, derleme önbelleği soğukken alınan değerdir. İkinci açılışta
`torch.compile` önbellekten geleceği için toplam düşecektir;
"sıcak derleme" cold start değeri ayrıca ölçülecektir.

---

## 4. Model davranışı

LFM2.5-2.6B saf bir reasoning modelidir ve her cevaptan önce düşünür.
Chat template `<think>` etiketini prompt'un sonuna eklediği için model
açılış etiketini üretmez; yalnızca `</think>` kapanış etiketi çıktıda
görünür.

Gözlem: "C. elegans nedir? Kısaca açıkla" sorusuna verilen cevapta
toplam 727 token üretildi; bunun yaklaşık 500'ü düşünme trace'i,
~220'si görünür cevaptı.

`max_tokens=200` ile yapılan ilk testte model `</think>`'e hiç
ulaşamadı ve `finish_reason: "length"` ile kesildi — çıktının tamamı
düşünme içeriğiydi.

**Rapor için sonuç:** Bu modelde "cevap başına toplam token" ile
"cevap başına görünür token" ayrı metriklerdir ve ikisi de
raporlanmalıdır. Capability testlerinde bu sabit düşünme maliyeti
gecikmeyi doğrudan etkileyecektir.

---

## 5. İlk performans referansı

Isınmış durumda, tek istek, 34 token prompt, 256 çıktı token
(`ignore_eos=true`):

| Metrik | Değer |
|---|---|
| ITL p50 / p95 / max | 9,4 / 9,6 / 10,8 ms |
| TPOT | 9,38 ms |
| Kullanıcı başına hız | 106,6 token/s |
| TTFT (kalıcı bağlantı) | ~17 ms |

**Roofline karşılaştırması:** RTX 4070 Ti SUPER'in bellek bant genişliği
~672 GB/s, model ağırlıkları 5,42 GB. Decode aşamasında her token için
tüm ağırlıklar okunduğundan teorik tavan:

```
672 GB/s ÷ 5,42 GB = ~124 token/s
```

Ölçülen 106,6 token/s, teorik tavanın **%86'sı**. Bu, decode
aşamasının bellek bant genişliğine bağlı olduğunun deneysel kanıtıdır
ve vLLM'in bu kartta neredeyse optimal çalıştığını gösterir. Kalan
%14 sampling, scheduler ve kernel launch payıdır.

ITL dağılımının darlığı (p50 ile max arasında 1,4 ms) termal throttle
veya scheduler tekleme olmadığını gösteriyor.

> Bu sayılar tek bir koşudan gelmektedir ve ön referans niteliğindedir.
> Resmî T1 sonuçları için tekrarlı ölçüm gerekir (METHODOLOGY 7).
