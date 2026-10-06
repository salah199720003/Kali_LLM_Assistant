# Qwen IQ3_XXS tuning — 2 October 2026

Historical results at 65,536 context. The current launcher uses 96,000 context
and disables MTP after a later latency check; see `iq3_latency_fix.md`.

Hardware: RTX 5070 12 GB, Ryzen 5 5600, 32 GB RAM. Runtime: llama.cpp build
11297 (`ca2e2037b`). Model: the existing Qwen3.8-27B-UD-IQ3_XXS.gguf. All trials
used a 65,536 context window and the same fixed synthetic prompt, tool schema,
seed, and sampling settings. Thinking was disabled for these timing probes.
No Kali commands were executed. Normal agent reasoning controls retain their
existing behavior.

## Applied changes

1. The terminal agent now puts changing controller state at the end of each
   request. Its system prompt and tool definitions remain a stable prefix.
   Evidence, budgets, and command checks still run in the controller.
2. The XXS launcher uses a logical batch of 512 and a physical batch of 128.
   This reduces temporary processing buffers while retaining the original
   model, Q4_0 cache, CPU cache placement, MTP with two draft tokens, and context.

## Measurements

These are individual local measurements, not averages or guarantees. Generation
rate is the server's `predicted_per_second`, excluding input processing. Input
processing time is the server's `prompt_ms`. Different reply lengths and GPU
competition affect total request time.

| Configuration | Cold input processing, 6,552 tokens | Generation, 128 tokens |
|---|---:|---:|
| Original CPU cache + MTP | 11.06 s | 14.39 tokens/s |
| GPU cache, 18 FFN blocks on CPU | 13.02 s | 9.50 tokens/s |
| Original + GPU sampling | 90.62 s | 15.34 tokens/s |
| CPU cache, 8 FFN blocks on CPU, smaller batches | 17.57 s | 9.37 tokens/s |
| Original without MTP | 7.37 s | 11.71 tokens/s |
| Original + batches 512/128 | 17.04 s | 14.09 tokens/s |

The cache test with batches 512/128 used 6,556 input tokens and a 48-token
response limit:

| Request | Cached input tokens | Input processing | Generation rate | Whole request |
|---|---:|---:|---:|---:|
| First request with stable layout | 0 | 16.73 s | 15.80 tokens/s | 19.72 s |
| Controller update with stable layout | 6,536 | 0.53 s | 15.18 tokens/s | 3.66 s |

The second request reused 99.7% of its input. With the old layout, changing the
controller state in the system prompt caused all 6,552 tokens to be reprocessed
in the original-runtime test, taking 10.56 seconds before generation.

A later reload of the original settings, while another game (MHUR.exe) was
using GPU memory, took 131.11 seconds to read 6,556 tokens and generated at
5.55 tokens/s. GPU memory was nearly full. This demonstrates why an isolated
fast run is insufficient evidence of stable performance with competing apps.
The smaller batches provide more headroom, but games can still exhaust VRAM.

## Validation and reproduction

All 246 offline agent regression tests passed after the request-layout change.
The tests cover the preserved system prefix, controller updates, tool results,
verification, scope checks, and reasoning controls. The launcher passed the
PowerShell syntax check.

The benchmark program is `experiments/tune_iq3.py`. It starts one temporary
server at a time on localhost port 8084 and stops it when each trial finishes.
Raw requests, timing responses, runtime arguments, and logs are saved in
`runtime/iq3-tuning/`. The old production server logs were preserved there
before bringing the normal endpoint back online.

Use `launch-27b.cmd` to load the updated terminal agent. Its endpoint remains
localhost port 8081. The separate Q4_K_S launcher retains its 98,304 context.
