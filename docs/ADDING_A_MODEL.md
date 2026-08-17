# Adding a model

The measurement procedure never changes between models. What changes are a
handful of model-specific constants and the limits of the T3 sweep. This
document covers both, plus what to expect from architectures that differ from
the one already measured.

Read [METHODOLOGY.md](METHODOLOGY.md) first if you have not — the reasoning
behind each server flag matters more than the flags themselves.

---

## Switching the model

The scripts read the model id and server URL from the environment, so no file
needs editing to point them at a different model:

```bash
export LLMBENCH_MODEL="google/gemma-3-4b-it"
export LLMBENCH_BASE_URL="http://127.0.0.1:8000/v1"   # optional
```

Without these the scripts default to `LiquidAI/LFM2.5-2.6B`. The server must
be serving the same model — vLLM returns 404 if the ids disagree.

---

## Procedure

### 1. First boot: use `--gpu-memory-utilization`, not a byte count

Sizing the KV cache in bytes requires knowing how much VRAM the weights
occupy, and that is only known after the model loads. So the first boot uses
the percentage form; the byte count is pinned from the second boot onward.

```bash
vllm serve <MODEL> \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching --seed 0 \
  2>&1 | tee docs/boot-<model>.log
```

Keep the log. It is the only source for several numbers that appear in the
report and cannot be measured any other way.

### 2. Read the startup log

```bash
grep -E "Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency|torch.compile|init engine" docs/boot-<model>.log
```

For LFM2.5-2.6B this produced:

```
Model loading took 5.05 GiB memory and 3.515591 seconds
Available KV cache memory: 7.46 GiB
GPU KV cache size: 488,404 tokens
Maximum concurrency for 32,768 tokens per request: 14.90x
torch.compile took 9.00 s in total
init engine (profile, create kv cache, warmup model) took 66.09 s
```

The first three are the inputs to everything below. The last two are the cold
start figures for the report.

### 3. Derive the per-token KV cost

Divide available KV cache memory by the reported token capacity:

```
7.46 GiB / 488,404 tokens ≈ 16,400 bytes per token
```

Weight size in GB (for the roofline calculation) is the reported GiB × 1.074:

```
5.05 GiB ≈ 5.42 GB
```

### 4. Update the constants in `analyze_context.py`

```python
WEIGHTS_GB = 5.42            # from step 3
KV_BYTES_PER_TOKEN = 16_400  # from step 3
BANDWIDTH_GB_S = 672.0       # property of the GPU, not the model
```

**This step fails silently if skipped.** No error is raised; the
predicted-vs-measured table is simply wrong, and it looks plausible. If you do
nothing else on this list, do this one.

### 5. Re-size the KV cache and the T3 sweep

Pin the KV cache in bytes so capacity stays identical across runs regardless of
what else is using VRAM at server start:

```
kv_cache = total × 0.85
         − VRAM used by other processes
         − weights
         − ~1 GiB vLLM overhead (activations, CUDA graphs, non-torch)
         − safety margin
```

Worked example, 16 GiB card, LFM2.5-2.6B:

```
13.59 − 1.04 − 5.05 − 0.95 ≈ 6.5 GiB   →  --kv-cache-memory=6442450944 (6 GiB)
```

A larger model leaves less. An 8 GB weight set on the same card gives roughly
3.6 GiB, so `--kv-cache-memory=3221225472`.

Then find the longest context that still fits:

```
kv_cache_bytes / bytes_per_token = maximum context in tokens
```

If the server reports `Maximum concurrency for N tokens per request` below
1.0, that context length does not fit at all and the T3 sweep must stop
earlier. Also remember that input plus output must fit the context window —
with `--osl 128` the largest usable input on a 131,072-token window is
130,944.

### 6. Run

T1 and T2 need no changes. T3's sweep comes from step 5.

```bash
# T1 — single-stream latency
python scripts/bench_latency_grid.py --scenario T1 \
  --isl 128,512,1024,2048,4096,8192 --osl 128,512 \
  --n 10 --warmup 5 --cooldown 5
python scripts/analyze_grid.py --plot

# T2 — concurrency
python scripts/bench_concurrency.py \
  --concurrency 1,2,4,8,16,32,64 --isl 1024 --osl 256 \
  --n-per-worker 5 --warmup 2 --cooldown 15
python scripts/analyze_concurrency.py --plot

# T3 — context scaling (restart the server with a larger --max-model-len)
python scripts/bench_latency_grid.py --scenario T3 \
  --isl <sweep from step 5> --osl 128 \
  --n 8 --warmup 3 --cooldown 10
python scripts/analyze_context.py --plot
```

Before trusting a full run, check stability first:

```bash
python scripts/probe_stability.py --n 20 --max-tokens 256
```

A TPOT coefficient of variation much above 0.5% means something else is
competing for the GPU. Close it before collecting data.

---

## What to expect from different architectures

The procedure is identical everywhere, but the *predictions* are not. The
decode model used in this repository —

```
slowdown ≈ (context × KV bytes per token) / weight size
```

— assumes the KV cache grows linearly with context and that every decoded token
reads the entire weight set. Both assumptions have exceptions.

| Architecture | KV cache behaviour | What to expect |
|---|---|---|
| Dense attention (Llama, Qwen) | High per token, linear | Quadratic prefill term appears earlier than 16k; lower context capacity |
| Hybrid convolution + attention (LFM2.5) | Low per token, linear | Measured here: quadratic term emerges past 16k, ratio 0.87–0.91 |
| Sliding window (Gemma 3) | Stops growing past the window on local layers | The linear model over-predicts at long context |
| Mixture of experts | Normal, but only active parameters are read per token | Compute the roofline from *active*, not total, parameters |
| Compressed KV / MLA (DeepSeek) | Very low per token | Unusually large context capacity |

Two of these deserve emphasis.

**Sliding window.** Only the global attention layers keep accumulating KV cache
past the window size; local layers cap out. So `KV_BYTES_PER_TOKEN × context`
overstates the real cache size at long context, and the model will predict more
slowdown than you measure. The divergence should begin around the window size.

**Mixture of experts.** The roofline ceiling is `bandwidth ÷ active parameter
bytes`, not total. A 30B model with 3B active reads roughly a tenth of its
weights per token and will decode far faster than its parameter count suggests.
Using the total will make the model look like it exceeds its own theoretical
ceiling, which is a sign the wrong number was used.

**A model that breaks the prediction is a result, not a problem.** Documenting
where the model fails and why is more informative than another table of numbers
that happen to fit. The LFM2.5 report states the bandwidth model is
systematically 11% pessimistic and offers a reason; a sliding-window model
diverging at its window boundary would be a stronger finding still.

---

## Keeping results comparable

Whatever changes between models, these should not:

- Same `--no-enable-prefix-caching`, same `--seed 0`, same `--dtype`
- Same ISL/OSL grid for T1 and same workload for T2 (ISL 1024, OSL 256)
- Same `ignore_eos: true` and `temperature: 0`
- Same warm-up count and same number of measured requests per cell

When something must differ — a smaller KV cache because the weights are larger,
a shorter T3 sweep because the context does not fit — record it in the model's
report rather than quietly changing it. Every run manifest already captures the
model id, package versions, GPU state and all parameters, so the raw data
remains self-describing either way.

Once two or three models have been measured, add a summary table to the README:
decode tok/s, fraction of roofline, KV capacity, and sustainable concurrency
under the SLO. One model is a measurement; three are a finding.
