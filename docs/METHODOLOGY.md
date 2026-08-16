# Metodoloji

Bu dosya, ölçümlerin nasıl yapıldığını ve her kararın neden verildiğini
tanımlar. Tüm model raporları bu dosyaya atıf yapar.

Son güncelleme: 2026-08-15

---

## 1. Test ortamı

| Bileşen | Değer |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| VRAM | 16.376 MiB |
| Güç limiti | 295 W (referans TGP'nin üzerinde, fabrika OC) |
| Sürücü | 591.86 (WSL passthrough, nvidia-smi 590.57) |
| OS | Windows + WSL2 → Ubuntu 26.04 LTS |
| Kernel | 6.18.33.2-microsoft-standard-WSL2 |
| Python | 3.12.14 (uv ile yönetiliyor) |
| gcc | 15.2.0 |
| vLLM | 0.26.0 |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| transformers | 5.15.0 |
| openai (client) | 3.1.0 |

Boştaki VRAM kullanımı (Windows masaüstü + Xwayland): ~1.068 MiB.
Bu değer değişkendir; her koşudan önce kaydedilir.

---

## 2. Metrik tanımları

**TTFT (Time To First Token)** — İsteğin gönderildiği andan, içerik taşıyan
ilk chunk'ın alındığı ana kadar geçen süre. `time.perf_counter()` ile
client tarafında ölçülür.

**ITL (Inter-Token Latency)** — İçerik taşıyan ardışık iki chunk arasındaki
süre. Bu kurulumda bir chunk = bir token olduğu doğrulanmıştır (bkz. 4.1),
dolayısıyla ITL = token başına süre.

**TPOT (Time Per Output Token)** — ITL değerlerinin aritmetik ortalaması.
Eşdeğer olarak `(E2E − TTFT) / (çıktı_token − 1)`.

**E2E gecikme** — İstek gönderiminden son chunk'ın alınmasına kadar geçen
toplam süre.

**Kullanıcı başına çıktı hızı** — `1000 / TPOT` (token/s). Tek bir
kullanıcının hissettiği akış hızı.

**Sistem çıktı verimi** — Tüm eşzamanlı isteklerin toplam çıktı token'ı /
duvar saati süresi. Sunucunun kapasitesi. Kullanıcı başına hızla ters
çalışır.

**Persentiller** — Tüm gecikme metrikleri p50/p95/p99 olarak raporlanır.
Yalnızca ortalama raporlamak, uzun kuyruğu gizlediği için yeterli değildir.

---

## 3. Token sayımı

Token sayıları **sunucudan** alınır, client tarafında tokenizer ile
sayılmaz. Streaming isteklerinde `stream_options={"include_usage": True}`
ile son chunk'taki `usage` bloğu kullanılır.

Gerekçe: her modelin tokenizer'ı farklıdır ve client tarafında sayım
modeller arası karşılaştırmada sistematik hata üretir.

---

## 4. Doğrulanmış varsayımlar

Aşağıdaki maddeler varsayım değil, bu donanım ve yazılım kombinasyonunda
ölçülerek doğrulanmıştır.

### 4.1 Bir chunk = bir token

vLLM'in OpenAI uyumlu streaming çıktısında akış şu yapıdadır:

```
chunk 0 : role="assistant", content=""     ← açılış, token taşımaz
chunk 1 : content="The"                    ← gerçek token
...
chunk N : content="Mer"                    ← gerçek token
chunk N+1: choices=[], usage={...}         ← usage chunk'ı
```

`max_tokens=5` ile yapılan testte tam olarak 5 içerik chunk'ı gözlendi.
256 token'lık bir istekte client sayacı 256, sunucunun `completion_tokens`
değeri 256 çıktı.

**Kural:** Rol alanı dolu ve içerik boş string olan chunk atlanır.
Yalnızca `content == ""` kontrolü yetersizdir (gerçek bir token da boş
string'e çözümlenebilir); yalnızca `role` kontrolü de yetersizdir
(bazı sunucular her chunk'ta rol gönderebilir). İkisi birlikte kontrol
edilir.

### 4.2 Client ek yükü ayrıştırıldı

TTFT ölçümlerinde üç ayrı gecikme katmanı tespit edilmiştir:

| Kaynak | Maliyet | Ne zaman |
|---|---|---|
| Süreç başlatma + HTTP bağlantı kurulumu | ~170 ms | Her yeni Python süreci |
| Sunucu soğuk kod yolları | ~430 ms | Yalnızca ilk istek |
| Gerçek TTFT (34 token prefill + localhost HTTP) | ~17 ms | Isınmış, kalıcı bağlantı |

Ölçüm kanıtı:

- Ayrı süreçlerde 8 ardışık koşu: 488, 179, 204, 178, 198, 192, 184, 199 ms
  → ~190 ms platosu (süreç + bağlantı maliyeti kalıcı)
- Tek süreçte 10 ardışık istek: 617, 17.2, 18.1, 16.5, 18.3, 17.5, 16.8,
  16.2, 17.4, 17.2 ms → ~17 ms platosu

**Kural:** Tüm istekler tek bir uzun ömürlü `AsyncOpenAI` örneği üzerinden
gönderilir. Süreç başına bir kez oluşturulur.

**Not:** Küçük ISL değerlerinde raporlanan TTFT, ~17 ms'lik harness taban
gecikmesini içerir. Bu değer her rapor sürümünde yeniden ölçülür.

### 4.3 Warm-up gerekçesi

İlk istek ile ikinci istek arasında TTFT farkı 36 kattır (617 → 17 ms).
İkinci istekten itibaren sistem kararlıdır.

**Kural:** Her sunucu başlatmasından sonra 5 warm-up isteği gönderilir.
Warm-up sonuçları hiçbir ortalamaya, persentile veya grafiğe dahil
edilmez. Model yeniden yüklendiğinde warm-up tekrarlanır.

Warm-up promptları test promptlarından farklıdır (prefix cache
kirlenmesini önlemek için).

---

## 5. Sunucu yapılandırması

### 5.1 Standart serve komutu

```bash
vllm serve <MODEL> \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --kv-cache-memory=6442450944 \
  --no-enable-prefix-caching \
  --seed 0
```

### 5.2 Flag gerekçeleri

**`--kv-cache-memory` (`--gpu-memory-utilization` yerine)**
`gpu-memory-utilization` toplam VRAM'in yüzdesini hedefler ve KV cache
boyutu sunucunun başlatıldığı andaki serbest belleğe göre hesaplanır.
Masaüstünde tarayıcı açıkken başlatılan sunucu, kapalıyken başlatılandan
farklı kapasiteye sahip olur — yani kapasite koşudan koşuya değişir.
Bayt cinsinden sabitlemek bu değişkenliği ortadan kaldırır ve manifest'e
tek bir sabit sayı olarak girer.

**`--no-enable-prefix-caching`**
Prefix caching vLLM V1'de varsayılan olarak açıktır. Aynı prompt tekrar
gönderildiğinde prefill atlanır ve TTFT sahte şekilde düşük ölçülür.
Latency, throughput ve context testlerinde kapalıdır.
Prefix caching ayrı bir deney ekseni olarak (cache hit vs miss TTFT farkı)
ileride ölçülecektir.

**`--seed 0`**
Tekrarlanabilirlik.

**Reasoning parser kullanılmaz**
`--reasoning-parser` düşünme içeriğini `reasoning_content` alanına ayırır
ve streaming chunk sayma mantığını değiştirir. Inference testlerinde ham
token akışı ölçüldüğü için etkinleştirilmez.

### 5.3 Çıktı uzunluğu kontrolü

Kontrollü testlerde `max_tokens` sabit tutulur ve `ignore_eos: true`
kullanılır. Aksi halde modeller farklı uzunlukta çıktı üretir ve
throughput sayıları farklı rejimlerden gelir, karşılaştırılamaz.

---

## 6. Ortam kararlılığı

WSL2'de `nvidia-smi -lgc` ile saat kilitleme çalışmaz. Bu nedenle kontrol
yerine gözlem stratejisi izlenir:

- Koşu boyunca 1 Hz ile `clocks.sm`, `temperature.gpu`, `power.draw`,
  `memory.used` loglanır ve ham sonuçların yanına kaydedilir.
- Koşular arasına 60-90 s cooldown eklenir.
- Ciddi ölçüm turlarında Windows masaüstü boş tutulur (tarayıcı kapalı).
  Tarayıcının GPU süreci hem VRAM tüketir hem de p95/p99'u kirletebilir.
- Her koşudan önce boştaki VRAM kaydedilir (temiz ortam ~1.068 MiB).

---

## 7. Tekrar ve istatistik

- Her konfigürasyon en az 3 kez, **sunucu yeniden başlatılarak** koşulur.
  Koşu-içi varyans ile koşu-arası varyans farklı şeylerdir; ikincisi
  genelde daha büyüktür ve çoğu benchmark bunu raporlamaz.
- Medyan ve IQR raporlanır.
- Ham veri (istek başına bir JSONL satırı) `results/raw/` altında saklanır
  ve asla silinmez. İşlenmiş özetler bundan yeniden üretilebilir.

---

## 8. Test kategorileri

| Kod | Test | Değişken | Sabit |
|---|---|---|---|
| T0 | Kapasite envanteri | — | — |
| T1 | Tek akış gecikmesi | ISL × OSL | concurrency = 1 |
| T2 | Eşzamanlılık doyma eğrisi | concurrency | ISL=1024, OSL=256 |
| T3 | Context ölçekleme | ISL | concurrency = 1, OSL=128 |

Capability testleri ikinci aşamadır ve bu dosyanın kapsamı dışındadır.
