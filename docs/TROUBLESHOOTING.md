# Kurulum sorunları ve çözümleri

WSL2 + Ubuntu 26.04 + RTX 4070 Ti SUPER ortamında karşılaşılan sorunlar.

---

## 1. Triton derleme hatası — motor hiç başlamıyor

**Belirti**

`vllm serve` çalıştırıldığında uzun bir traceback ve sonunda:

```
torch._inductor.exc.InductorError: CalledProcessError: Command
'['/usr/bin/gcc', '/tmp/.../cuda_utils.c', '-O3', '-shared', ...]'
returned non-zero exit status 1
...
RuntimeError: Engine core initialization failed.
```

Traceback uzun ama kök neden en alttaki tek satırdadır: gcc çağrısı
başarısız olmuştur.

**Sebep**

vLLM açılışta `torch.compile` ile kernel üretir; bu Triton'a bağımlıdır.
Triton ilk iş olarak kendi CUDA yardımcı modülünü (`cuda_utils.c`)
derlemeye çalışır. Bunun için Python geliştirme başlıklarına ihtiyaç
duyar. Ubuntu 26.04 varsayılan olarak Python 3.14 ile gelir ve
`python3-dev` kurulu değildir, dolayısıyla `Python.h` yoktur.

**Kontrol**

```bash
ls -l /usr/include/python3.14/Python.h
```

"No such file or directory" → teşhis doğrulanmıştır.

**Çözüm (tercih edilen)**

Python 3.12'ye geçmek. Yalnızca başlık dosyalarını kurmak seni
Python 3.14 + gcc 15 kombinasyonunda bırakır; ikisi de bu yığın için
oldukça yenidir ve bu ilk çatlak muhtemelen tek çatlak olmayacaktır.

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
sudo apt install -y build-essential
```

uv'nin getirdiği Python kendi başlık dosyalarını taşır, ayrıca
`python3-dev` gerekmez.

**Alternatif çözüm (3.14'te kalarak)**

```bash
sudo apt install -y build-essential python3-dev python3.14-dev
```

**Doğrulama**

```bash
python -c "
import sysconfig, os
print('Python.h:', os.path.exists(
    os.path.join(sysconfig.get_paths()['include'], 'Python.h')))
import triton, torch
print('triton', triton.__version__, '| torch', torch.__version__)
print('cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))
"
```

---

## 2. uv kurulduktan sonra PATH'e girmiyor

**Belirti**

```
Command 'uv' not found, but can be installed with:
sudo snap install astral-uv
```

Bu sinsi bir hatadır çünkü sessizce başarısız olur: `uv pip install`
komutları hiçbir şey kurmadan geçer, sonra ilgisiz görünen bir
`ModuleNotFoundError` alırsın.

**Sebep**

Installer PATH'i mevcut kabuğa eklemez. `source ~/.bashrc` de yeterli
değildir çünkü satır henüz `.bashrc`'de yoktur.

**Çözüm**

```bash
source $HOME/.local/bin/env
echo '. "$HOME/.local/bin/env"' >> ~/.bashrc
which uv && uv --version
```

**Ders**

Toplu komut yapıştırdıktan sonra çıktıyı satır satır kontrol et.
"command not found" satırları akışın ortasında kolayca gözden kaçar.

---

## 3. Yanlış terminalde çalışma

**Belirti**

```
Command 'python' not found, did you mean: command 'python3' from deb python3
grep: docs/server-boot.log: No such file or directory
```

**Sebep**

Yeni açılan terminalde venv aktif değil ve/veya proje dizininde değilsin.
Sunucu bir terminalde çalışırken ikinci terminal açmak gerektiğinde sık
karşılaşılır.

**Çözüm**

```bash
cd ~/local-llm-benchmark
source .venv/bin/activate
```

Prompt `(local-llm-benchmark)` ile başlamalıdır.

---

## 4. Client kapanışında async generator hatası

**Belirti**

```
RuntimeError: generator didn't stop after athrow()
```

`await client.close()` sonrasında görülür.

**Sebep**

Kapanış anında henüz tamamen tüketilmemiş bir streaming generator vardır;
httpcore2 temizlenirken şikâyet eder.

**Etki**

Ölçümü etkilemez — tüm sayılar bu hata oluşmadan önce hesaplanmıştır.

**Çözüm**

Client'ı `async with` bloğuyla yönet ve stream'leri açıkça sonuna kadar
tüket.

---

## 5. openai SDK: boş api_key reddediliyor

**Belirti**

```
openai.OpenAIError: Missing credentials. Please pass an `api_key` ...
```

**Sebep**

openai Python SDK v2.34.0'dan itibaren `api_key=""` kabul edilmiyor.
Kimlik doğrulaması istemeyen yerel sunucularda (vLLM, llama.cpp,
LM Studio) yaygın olarak boş string geçiliyordu.

**Çözüm**

Dolu ama anlamsız bir değer geç:

```python
client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
```

---

## 6. Kurulum sırası: transformers'ı vLLM'den sonra kur

LFM2.5 tokenizer'ı `transformers>=5.0.0` gerektirir. vLLM'in bağımlılık
çözücüsü transformers'ı geri düşürebilir, bu yüzden sıra önemlidir:

```bash
uv pip install vllm==0.26.0
uv pip install -U "transformers>=5.0.0"   # vLLM'den SONRA
uv pip install openai
```
