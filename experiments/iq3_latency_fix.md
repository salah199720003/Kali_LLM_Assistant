# XXS input-processing delay — 3 October 2026

Hardware: RTX 5070 12 GB, Ryzen 5 5600, 32 GB RAM. Runtime: b11297
(`ca2e2037b`). Existing IQ3_XXS model, **96,000 context**, Q4_0 CPU cache,
Flash Attention, one slot. Thinking was disabled. Current desktop applications
remained open. Temporary servers used port 8084 and stopped after each probe.
No Kali commands were executed. This check did not resume the canceled model
or workflow-quality comparison.

## Finding and change

The reported 96-second wait was input processing, before token generation.
The recorded production run processed only 1,024 tokens in 79.26 seconds.
Its first greeting spent 26.00 seconds processing 189 input tokens, then
0.48 seconds generating ten output tokens.

A direct request containing only a short system message and `hello` reproduced
the delay without the terminal controller. This rules out the controller's
request-layout change as a sufficient explanation for the delay.

MTP added approximately 334.75 MiB of GPU model weights and a 415.44 MiB GPU
compute buffer. Its additional memory demands were associated with severe
input-processing delays. Windows GPU counters showed roughly 658–718 MiB of
shared usage with MTP, compared with about 110 MiB without MTP. GPU power during
the slow input runs was about 33 W, versus about 108 W during the faster run.
These observations support memory pressure and driver fallback as the cause;
shared-memory counters alone do not distinguish every CUDA allocation type.

The XXS launcher now explicitly uses `--spec-type none`. The context remains
96,000, batches remain 512/128, and model and cache precision are unchanged.
Agent request sampling and the 8,192-token thinking ceiling are unchanged.
Browser reasoning and sampling controls remain separate. The separate Q4
launcher is unchanged. The shared XXS server uses standard decoding for all
clients connected to it.

## Local measurements

The short request contained 24 input tokens and generated ten output tokens.

| Runtime | Short-request input processing | Output tokens/s | Agent input check |
|---|---:|---:|---|
| Existing MTP, 512/128 | 15.70 s | 4.23 | Production logs already show the delay |
| MTP, CUDA graphs disabled, 512/128 | 11.23 s | 23.77 | Did not finish within 60 s |
| MTP, 128/32 | 12.53 s | 22.92 | Did not finish within 60 s |
| Standard decoding, 512/128 | 0.46 s | 14.15 | Complete response; details below |

The final check captured a request through the actual terminal controller using
the existing offline test harness, with sudo mode enabled. It sent the captured
instructions and tool schema to the temporary server, but never executed the
returned command. The output ceiling was 512 tokens to bound the diagnostic;
the agent's real 32,768-token output ceiling was not changed.

| Request | Input tokens processed | Cached tokens | Input processing | Output tokens | Output tokens/s | Whole request |
|---|---:|---:|---:|---:|---:|---:|
| Cold agent IP question | 5,748 | 0 | 9.03 s | 86 | 12.52 | 15.91 s |
| Identical repeated question | 4 | 5,744 | 0.14 s | 112 | 13.17 | 8.59 s |

Both responses completed as tool calls; their argument JSON parsed successfully.
This verifies generation and cache reuse, not SSH execution, command admission,
or end-to-end answer quality. The diagnostic request did not include the user's
previous connection and sudo conversation, so it was shorter than the reported
production request. The first standard-decoding probe also read 5,478 tokens in
7.98 seconds, but its 64-token diagnostic ceiling truncated the tool call; that
probe is evidence for input speed only.

MTP can improve generation speed when its extra allocations fit. Disabling it
can reduce output tokens/s compared with a healthy MTP run. Here, it removed
the large wait before generation while preserving the existing model and 96K
context. Longer contexts and competing GPU applications still need enough
memory; this short check does not prove performance with a full context.

Earlier offline regression tests covered controller behavior and did not prove
GPU latency or model-answer equivalence. Earlier cache figures came from 65,536
context and follow-up requests with a stable prefix. They did not establish that
a first shell request or a switch from chat to shell would process quickly at 96K.

Raw arguments, model responses, runtime logs, and GPU counters are saved under
`runtime/prefill-delay-*`; the final complete-response probe is
`runtime/prefill-delay-20261003-025835/`. The diagnostic script is
`runtime/check_prefill_delay.py`.
