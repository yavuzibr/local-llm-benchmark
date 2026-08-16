# Methodology

This document defines how the measurements are performed and why each decision was made.
All model reports reference this document.

Last updated: 2026-08-15

---

## 1. Test Environment

| Component       | Value                                       |
| --------------- | ------------------------------------------- |
| GPU             | NVIDIA GeForce RTX 4070 Ti SUPER            |
| VRAM            | 16,376 MiB                                  |
| Power limit     | 295 W (above reference TGP, factory OC)     |
| Driver          | 591.86 (WSL passthrough, nvidia-smi 590.57) |
| OS              | Windows + WSL2 → Ubuntu 26.04 LTS           |
| Kernel          | 6.18.33.2-microsoft-standard-WSL2           |
| Python          | 3.12.14 (managed with uv)                   |
| gcc             | 15.2.0                                      |
| vLLM            | 0.26.0                                      |
| PyTorch         | 2.11.0+cu130                                |
| Triton          | 3.6.0                                       |
| transformers    | 5.15.0                                      |
| openai (client) | 3.1.0                                       |

Idle VRAM usage (Windows desktop + Xwayland): ~1,068 MiB.
This value is variable and is recorded before every run.

---

## 2. Metric Definitions

**TTFT (Time To First Token)** — The time elapsed from when the request is sent until the first content-bearing chunk is received. Measured on the client side using `time.perf_counter()`.

**ITL (Inter-Token Latency)** — The time between two consecutive content-bearing chunks. In this setup, one chunk has been verified to correspond to one token (see 4.1), therefore ITL = time per token.

**TPOT (Time Per Output Token)** — The arithmetic mean of ITL values.
Equivalently: `(E2E − TTFT) / (output_tokens − 1)`.

**E2E latency** — The total time from sending the request until the final chunk is received.

**Per-user output rate** — `1000 / TPOT` (tokens/s). The streaming rate experienced by a single user.

**System throughput** — Total output tokens from all concurrent requests / wall-clock time. This represents server capacity and moves inversely to per-user speed.

**Percentiles** — All latency metrics are reported as p50/p95/p99. Reporting only the mean is insufficient because it hides the long tail.

---

## 3. Token Counting

Token counts are obtained **from the server**, rather than tokenizing on the client side. For streaming requests, the `usage` block in the final chunk is used with `stream_options={"include_usage": True}`.

Rationale: each model uses a different tokenizer, and client-side token counting can introduce systematic errors in cross-model comparisons.

---

## 4. Verified Assumptions

The following are not assumptions; they have been experimentally verified for this specific hardware and software configuration.

### 4.1 One Chunk = One Token

vLLM's OpenAI-compatible streaming output follows this structure:

```text
chunk 0 : role="assistant", content=""     ← opening, no token
chunk 1 : content="The"                    ← actual token
...
chunk N : content="Mer"                    ← actual token
chunk N+1: choices=[], usage={...}         ← usage chunk
```

A test with `max_tokens=5` produced exactly 5 content chunks.
For a 256-token request, the client-side counter reported 256 and the server's `completion_tokens` value was also 256.

**Rule:** A chunk with a populated role field and an empty content string is skipped. Checking only `content == ""` is insufficient because a real token may also resolve to an empty string; checking only `role` is insufficient because some servers may include the role in every chunk. Both conditions are checked together.

### 4.2 Client-Side Overhead Was Isolated

Three distinct latency layers were identified in TTFT measurements:

| Source                                          |    Cost | When                          |
| ----------------------------------------------- | ------: | ----------------------------- |
| Process startup + HTTP connection setup         | ~170 ms | Every new Python process      |
| Server cold code paths                          | ~430 ms | First request only            |
| Actual TTFT (34-token prefill + localhost HTTP) |  ~17 ms | Warmed, persistent connection |

Measurement evidence:

* 8 consecutive runs in separate processes: 488, 179, 204, 178, 198, 192, 184, 199 ms
  → ~190 ms plateau (process + connection overhead persists)
* 10 consecutive requests in a single process: 617, 17.2, 18.1, 16.5, 18.3, 17.5, 16.8, 16.2, 17.4, 17.2 ms
  → ~17 ms plateau

**Rule:** All requests are sent through a single long-lived `AsyncOpenAI` instance. One instance is created per process.

**Note:** For small ISL values, the reported TTFT includes the ~17 ms baseline latency of the harness. This value is re-measured for every report revision.

### 4.3 Warm-Up Rationale

The TTFT difference between the first and second request is 36× (617 → 17 ms). The system is stable from the second request onward.

**Rule:** After every server startup, 5 warm-up requests are sent. Warm-up results are excluded from all averages, percentiles, and graphs. Warm-up is repeated whenever the model is reloaded.

Warm-up prompts are different from test prompts to prevent prefix-cache contamination.

---

## 5. Server Configuration

### 5.1 Standard Serve Command

```bash
vllm serve <MODEL> \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --kv-cache-memory=6442450944 \
  --no-enable-prefix-caching \
  --seed 0
```

### 5.2 Flag Rationale

**`--kv-cache-memory` (`--gpu-memory-utilization` alternative)**

`gpu-memory-utilization` targets a percentage of total VRAM, and the KV cache size is calculated based on the amount of free memory available when the server starts.

A server started while a browser is open on the desktop therefore has a different capacity from one started with the browser closed — meaning capacity varies from run to run.

Fixing the value in bytes eliminates this variability and allows a single constant value to be recorded in the manifest.

**`--no-enable-prefix-caching`**

Prefix caching is enabled by default in vLLM V1. When the same prompt is sent again, prefill is skipped and TTFT is artificially reduced.

It is disabled for latency, throughput, and context tests.

Prefix caching will be measured separately as an experimental axis in the future (cache-hit vs. cache-miss TTFT).

**`--seed 0`**

For reproducibility.

**Reasoning parser is not used**

`--reasoning-parser` separates reasoning content into the `reasoning_content` field and changes the streaming chunk-counting behavior.

Since inference tests measure the raw token stream, it is not enabled.

### 5.3 Output Length Control

For controlled tests, `max_tokens` is kept fixed and `ignore_eos: true` is used.

Otherwise, models may generate outputs of different lengths, causing throughput numbers to come from different operating regimes and making comparisons invalid.

---

## 6. Environment Stability

Clock locking with `nvidia-smi -lgc` does not work under WSL2. Therefore, an observation-based strategy is used instead of clock locking:

* `clocks.sm`, `temperature.gpu`, `power.draw`, and `memory.used` are logged at 1 Hz throughout each run and stored alongside the raw results.
* A 60–90 s cooldown period is used between runs.
* During serious measurement rounds, the Windows desktop is kept idle (browser closed). The browser's GPU process can consume VRAM and contaminate p95/p99 measurements.
* Idle VRAM is recorded before every run (clean environment: ~1,068 MiB).

---

## 7. Repetition and Statistics

* Each configuration is run at least 3 times, **with the server restarted between runs**. Within-run variance and between-run variance are different things; the latter is generally larger and is not reported by many benchmarks.
* Median and IQR are reported.
* Raw data (one JSONL record per request) is stored under `results/raw/` and is never deleted. Processed summaries can be regenerated from the raw data.

---

## 8. Test Categories

| Code | Test                         | Variable    | Fixed                    |
| ---- | ---------------------------- | ----------- | ------------------------ |
| T0   | Capacity inventory           | —           | —                        |
| T1   | Single-stream latency        | ISL × OSL   | concurrency = 1          |
| T2   | Concurrency saturation curve | concurrency | ISL=1024, OSL=256        |
| T3   | Context scaling              | ISL         | concurrency = 1, OSL=128 |

Capability tests are a second-stage effort and are outside the scope of this document.
