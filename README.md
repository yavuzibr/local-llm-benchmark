# Local LLM Benchmark

Inference benchmarks for LLMs running on my consumer hardware, served with vLLM.
Every result comes with the raw per-request data and the run manifest that
produced it.

## Test Environment

**Hardware**

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| VRAM | 16,376 MiB |
| Memory bandwidth | ~672 GB/s (256-bit GDDR6X) |
| Power limit | 295 W |
| Driver | 591.86 (CUDA 13.1) |

**Software**

| | |
|---|---|
| OS | Windows + WSL2 → Ubuntu 26.04 LTS |
| Kernel | 6.18.33.2-microsoft-standard-WSL2 |
| Python | 3.12.14 (managed by uv) |
| gcc | 15.2.0 |
| Inference engine | vLLM 0.26.0 |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| transformers | 5.15.0 |
| openai (client) | 3.1.0 |

**Server configuration**

| | |
|---|---|
| Precision | BF16 (no quantization) |
| KV cache | pinned at 6 GiB (`--kv-cache-memory=6442450944`) |
| Prefix caching | disabled |
| Seed | 0 |


## Tests

| | Question it answers | Variable | Held fixed |
|---|---|---|---|
| **T0** Capacity | What can this card hold? | — | — |
| **T1** Single-stream latency | How fast does one user get a response? | ISL × OSL | concurrency = 1 |
| **T2** Concurrency | How many users can I serve at once? | concurrency | ISL=1024, OSL=256 |
| **T3** Context scaling | How much slower on long inputs? | ISL (1k → 128k) | concurrency = 1, OSL=128 |

Capability testing (reasoning, instruction following, tool use) is phase two.
## Models

- [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) — BF16

## Methodology

Full details and the reasoning behind every decision:
[docs/METHODOLOGY.md](docs/METHODOLOGY.md)

Four choices that determine whether the numbers mean anything:

**Prefix caching is disabled.** On by default in vLLM V1. With it on, repeated
prompts skip prefill and TTFT reads artificially low — the benchmark ends up
measuring its own cache hit rate. Prompts are also generated unique per request.

**Output length is pinned.** `max_tokens` fixed plus `ignore_eos=true`. Without
this, models produce different output lengths and throughput figures come from
different regimes, so they can't be compared.

**KV cache is pinned in bytes, not as a percentage.** `--gpu-memory-utilization`
sizes the cache from whatever VRAM is free at server start, so capacity drifts
between runs depending on what else is open. A fixed byte count removes that.

**Token counts come from the server**, via the streaming `usage` block — never
from client-side tokenization, which introduces systematic error across models.

## Measurement validation

The harness was calibrated before any result was trusted:

- **One chunk = one token**, verified against the raw SSE stream. Without this,
  ITL means "time per chunk," not "time per token."
- **Client overhead isolated.** Early TTFT readings were 465 ms for a 34-token
  prompt. Decomposition showed ~170 ms of process/connection setup, ~430 ms of
  one-time server warm-up, and ~17 ms of actual TTFT — 96% of the reported
  number was instrumentation.
- **Warm-up justified by data**, not convention: the first request costs 36×
  the second.
- **The client is not the bottleneck.** Two independent processes at N=64 each
  produced 2,470 tok/s against a single process at N=128 measuring 2,429 —
  a 1.7% difference.
- **Run-to-run variance measured at 0.046% CV** across three runs with full
  server restarts, below the 0.317% within-run variance.

Every number in the report is regenerable from `results/raw/`.

Benchmarking a different model: [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md)

