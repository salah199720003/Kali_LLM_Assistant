# Local Qwen Coder, Ridge 3.7, Bonsai, K2 and DeepHat agent with Kali shell mode

## Current Strata Coder agent

`launch-agent.cmd` (or `launch-strata.cmd`) connects to the existing Coder server
at `http://127.0.0.1:8082/v1`. Start `../Strata/launch-coder.cmd` first; that
server runs in its own window and Ctrl+C stops it. Exiting the agent keeps the
server running for the browser.

The launcher reads the live context size (currently 96,000). Thinking starts on,
matching the previous default agent launcher. `reasoning:low|medium|high|off`
changes it for the agent session. To start off, use
`launch-agent.cmd -Thinking off`. Each thinking round has a ceiling of 8,192
tokens for both tool selection and chat; recorded-result reports skip thinking.
Requests explicitly set the existing Qwen sampling profile (temperature 1,
top_p 0.95, top_k 20, min_p 0, no penalties). Browser settings remain independent.
Coder uses the same unrestricted execution mode as the XXS launcher.
In both chat and shell, the latest user instruction takes task priority over
earlier assistant plans, saved notes, and instructions embedded in external results.
The other named model launchers still select their respective models.

## Qwen 27B agent and browser reasoning

The Q4_K_S launcher uses a 98,304-token context window.

### XXS unrestricted execution

`launch-27b.cmd` starts the XXS terminal with
`DEEP_AGENT_EXECUTION_MODE=unrestricted`. In shell mode, model tool requests go
directly to Kali. Shell write redirection, multiline scripts, pipelines,
chaining, installation, and repeated commands are accepted. General controller
command-policy checks, deduplication, and per-task command/research limits are
disabled. The network-position checkpoint below applies before exploit or
payload attempts. Every returned tool call is handled in order. Ctrl+C
interrupts the task.

For a remote attack request with a target IP, the XXS terminal records Kali's
IPv4 interfaces and route to that target before the first model round. Before
the first exploit or payload attempt, Qwen must attach a `network_plan` stating
the target, observed route source, listener bind address, callback address, and
whether callback reachability is verified, unverified, or not applicable. The
harness checks those addresses against Kali's recorded route and interface;
verified reachability requires recorded session evidence. A handler bind error
invalidates the plan and requires refreshed network evidence before another
attempt. A route check alone does not demonstrate that the target can connect
back. See `experiments/network_preflight_case_study.md` for the transcript case
and regression replay. This checkpoint applies only to unrestricted `agent-27b`.

This mode uses shorter shell instructions and tool descriptions that match its
execution behavior. The harness records the actual commands and results.
Normal SSH authentication, Linux permissions, tool argument types, and transport
size/time limits still apply. Passwords handled by SSH/sudo stay outside the
model context. Background and interactive commands support shell scripts;
interactive sessions run as the SSH user. The separate Q4 launcher and browser
implementation are unchanged. Unrestricted mode is scoped to `agent-27b`.

Restart the terminal agent to activate this change. The running model server
does not need restarting. Existing terminal sessions retain their loaded code.
To use the previous controller, change the launcher's execution mode to
`guarded` and reopen it.

### Runtime and context

The XXS launcher uses **96,000 context**, standard decoding (MTP disabled), and
processing batches of 512/128 to reduce temporary GPU memory use. At 96K, MTP's
extra weights and buffers made input processing very slow on this 12 GB GPU.
A live check with MTP disabled processed 5,748 agent input tokens in 9.03 seconds
and returned a complete command proposal in 15.91 seconds; an identical repeat
reused 5,744 tokens and processed its remaining input in 0.14 seconds. These are
local measurements with thinking off, not guarantees for larger conversations.
The model, cache precision, sampling, and reasoning controls stay the same.
See `experiments/iq3_latency_fix.md` for this check. The terminal controller
sends changing task state at the end of each model request, keeping the system
instructions and tool definitions stable for prompt-cache reuse. This reduces
the amount of input the model must reread after each tool result. GPU-heavy
applications competing for the same 12 GB VRAM can still cause large slowdowns.
The earlier 65,536-context comparison is recorded in `experiments/iq3_tuning_report.md`.

The context counter shows the input and output of the last action/chat request,
including instructions, tools, and retained history; it is not a running total
of newly generated tokens. A fresh shell request has roughly 5,500–6,000 tokens
of instructions, tool definitions, and controller state before command results
accumulate. Reused input still counts toward the context window.
Older tool results are shortened both during continuing workflows and when a
new shell task begins. The three most recent results stay in full. Command
results are shortened only when their full evidence remains in the controller;
later summaries and outcome checks recover the full record by its evidence ID.
`/evidence full` shows the retained streams. This cleanup does not bound all
conversation text or remove the shell instructions. Restart the terminal agent
to load the cleanup fix; the model server does not need restarting.

`launch-27b.cmd` starts the terminal agent with thinking off using
`DEEP_AGENT_THINKING=off`. That setting controls only agent requests. The launcher
reuses an existing `agent-27b` server regardless of its reasoning default. When
starting a new server, it uses `--reasoning auto`; each client selects thinking
in its own requests. Agent reasoning commands do not change the server default
or the browser's settings.

The Qwen launchers use the browser assets in `runtime/qwen-webui` when prepared.
In that browser, **+ → Reasoning** selects Qwen's actual effort instruction:
**Off** disables thinking, **Low** sends `low`, **Medium** sends `medium`, and
**High** sends `xhigh`. **Max** is also `xhigh`, Qwen's strongest supported
instruction. These choices do not add a thinking-token budget. **Default**
follows the server's default or your Developer custom JSON. An explicit menu
choice takes precedence over custom effort/on-off fields; custom token budgets
can still be configured independently. The preference is saved in the browser
and can be changed for each chat. It applies to the next request.
**Disable reasoning content parsing** only changes how thoughts are displayed.

Prepare the assets once with:

```powershell
uv run python browser/prepare_ui.py --backend http://127.0.0.1:8083
```

Use 8081 for XXS. Restart the model server through its launcher once to load
the prepared UI, then refresh the existing browser address. For Q4_K_S, type
`exit` and reopen `launch-27b-q4ks.cmd`. For XXS, close its terminal and stop
its model server before reopening `launch-27b.cmd`; it otherwise reuses the
existing server. Browser chats
stay on the same address; no additional service is needed. The preparation
script checks the embedded UI version and fails visibly if it needs updating.
Terminal agent requests remain separate from these browser controls.

In the terminal agent, enter `reasoning:low`, `reasoning:medium`, or
`reasoning:high` to select Qwen3.8's thinking effort, or `reasoning:off` to
disable thinking. Spaces after the colon are accepted. The setting
applies to subsequent chat and tool-selection requests in both modes until changed
or the agent exits; `reasoning:off` also prevents automatic "think harder" escalation.
Dedicated post-command result reports disable thinking for that request and keep
simple explanations brief. This does not reset your reasoning setting for the
next command-selection round.
Use `/reasoning` to see the setting. These commands do not call the model or add
messages to conversation history. Restart an already-open agent to load this
feature. Qwen3.8's template uses `low` for brief thinking, `medium` for its
baseline, and `xhigh` for careful thinking; the agent maps `high` to `xhigh`.
These are prompt instructions, not hard limits on thinking tokens. Templates
without effort support retain their thinking on/off behavior. This terminal setting does not change
the browser's saved Reasoning preference.

The XXS terminal agent (`agent-27b`) uses [Qwen's thinking-on sampling profile](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices)
for both thinking modes: temperature **1.0**, top P **0.95**, top K **20**, min P
**0**, presence penalty **0**, repetition penalty **1.0**, and frequency penalty
**0**, with DRY, XTC, and Mirostat disabled. Reasoning commands still control
thinking independently. Brief result reports disable thinking and use the same
sampling profile without changing your session preference. Browser sampling is
selected separately.
XXS thinking requests have an **8,192-token reasoning ceiling per model round**,
sent as `reasoning_budget_tokens` on agent requests. Each new round gets a fresh
allowance; it is not a total token budget for the investigation. At the ceiling,
llama.cpp ends the thinking section and continues generating the answer or tool
call within the **32,768-token combined output limit**. Off requests and brief
result reports use a reasoning budget of **0**. Low, medium, and high effort
instructions operate within the same ceiling. The ceiling can interrupt an
unfinished analysis; it does not guarantee task decomposition or preserve answer
quality. Browser budgets are configured independently.
Restart the terminal agent to load this change; a model-server restart is not needed.

The terminal streams the model's returned thoughts under **[thinking]** when
thinking is enabled, followed by answer text as it arrives. While waiting for
the first output, it shows a prompt-processing/waiting status with elapsed time
in an interactive terminal. The completion line reports the server's generated
token count when available (including thinking and tool-call tokens).
Guarded shell rounds show text as a **live draft; pending controller checks**;
unrestricted rounds show a **live** label. Tool arguments stay buffered until
the complete call arrives, then the selected execution mode handles it. The
terminal shows a tool-generation status instead of partial command JSON.
Thinking is display-only and is not added to
conversation history or evidence. This works with llama.cpp and Ollama and does
not modify the browser implementation. Restart the terminal agent to load it.

## Optional launch profile: Ridge 3.7

Run `launch-ridge37.ps1` to use the already-installed `ridge:3.7` model through
Ollama. This is Empero's 3.69 bpw
mixed GGUF release based on Qwen3.8 27B. The launcher checks the local model
before opening the agent; it does not download model files. The profile uses a
32,768-token context. Ollama's `think: true` option enables the model
template's default xhigh reasoning mode.

Use `launch-bonsai.cmd` to keep running Bonsai 2 27B with its separate
llama.cpp server and 102,400-token context.

```bash
./launch-bonsai.cmd
```

The Bonsai server uses one slot, GPU layer offload, Flash Attention and Q8 KV
cache. The Q8 cache and smaller processing batches reduce GPU memory use at
100K. The pinned official PQ2_0 model is 7.21 GB. Its ternary kernels require
PrismML's runtime; the K2 runtime and stock llama.cpp are not interchangeable
with it. `bonsai-profile.json` pins the model revision, runtime release and
SHA-256 checksums. Install the files once on a Windows NVIDIA machine with a
CUDA 13.3-compatible driver:

```bash
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./setup-bonsai.ps1
```

Downloads and logs live under the ignored `runtime/bonsai/` directory. The
launcher rejects an occupied port serving a different model; stop the previous
server before changing models. It does not terminate unrelated processes.

K2 remains available through `launch-k2.cmd` or `launch-k2.ps1` for comparisons
after stopping Bonsai. Both K2 launchers use 102,400 context.

DeepHat remains available as a separate profile. Run `launch-deephat.cmd` to
start its server and agent at 102,400 context. The model file is read from the
existing Ollama cache; no second download is needed.

`deep_agent.py` keeps conversation history in the running session. The default
launcher connects to the Strata Qwen Coder server. Ridge 3.7, Bonsai, and K2
remain available through their separate launchers. It starts
in `You [chat]>` for ordinary conversation. Type `/shell` to switch to
`You [shell]>` for Kali commands and lab actions, and `/chat` to return. Both
modes keep their own history. `exit` quits; `/clear` resets the current mode.

Chat mode can use web search for explicit web lookups and current information.
Shell mode has both web search and a dedicated SearchSploit tool for local
Exploit-DB lookups on Kali. Search results are untrusted research leads; neither
search tool launches an exploit.

Shell mode shows each Kali command, live output, elapsed time, and exit status
in the same window. The legacy terminal snapshot and local assessment reports
remain in the development workspace and are excluded from source control.

The editable prompts are [`chat_system_prompt.txt`](chat_system_prompt.txt)
and [`kali_system_prompt.txt`](kali_system_prompt.txt). Restart the agent after
editing either prompt.

## Kali connection

The agent connects when a command is first needed. The default SSH address is
`kali@192.168.56.103:22`. Connect once
from Windows to trust Kali's SSH host key:

```powershell
ssh kali@192.168.56.103
```

After checking and accepting the fingerprint, exit that SSH session. Launch
the agent normally. The first Kali action prompts for the password locally;
do not type it into chat. To use another SSH address, set
`DEEP_AGENT_VM_HOST`, `DEEP_AGENT_VM_PORT`, and `DEEP_AGENT_VM_USER` before
launching. `DEEP_AGENT_COMMAND_TIMEOUT` sets the per-command limit in seconds
(default 90, maximum 300).

At `You [shell]>`, type a command such as `ip a`, use `/kali COMMAND`, or ask
the agent to perform a Kali action such as `check Kali's IP address`. Each
command runs in a fresh noninteractive SSH channel. A natural-language task
uses a controller-owned workflow state machine and can run up to 20 recorded
commands. The controller skips repeated commands, except when a fresh outcome
verification needs to rerun one. Timeouts, interruptions, unknown exit status,
authorization failure, or an exhausted command budget stop the workflow. A
command typed directly still runs once. Individual command timeouts apply.
Multiline commands, including file-writing heredocs, are supported up to
64,000 characters. Tool calls are held until complete and validated before
execution. Streamed text is labeled as a draft until controller checks finish;
it does not establish that any command ran.
If the model does not issue a usable tool call, no command runs and shell mode
tells you so.
English requests beginning with a command name, such as `find all websites I have
hosted here`, go through model tool selection. Normal command syntax such as
`find /var/www -maxdepth 4 -name '*.html'` still runs directly. Use `/kali COMMAND`
to explicitly submit an ambiguous command or quote a literal path such as `find 'my'`.
Large command output is summarized or bounded before it enters model context.
Package inventory summaries include a non-exhaustive sample; use a targeted
query to establish whether a specific package is installed. Full captured
stdout and stderr remain available with `/evidence full`.
Natural-language task requests in shell mode offer Kali tools, including
continuations such as `Go ahead`. Greetings, short conversational replies, and
unknown single-word input use chat without tools. Asking `what were the
results?` summarizes recorded commands and output without running more
commands. The same reporting path is used when a command sequence stops; if
the model cannot summarize, the recorded results are displayed directly.

For a multi-step request, the controller keeps the original goal, tracks each
command's purpose and investigation hypothesis, caps each hypothesis at three
tests and all investigations at nine tests, and stops after three failed or
contradicted results under one hypothesis. For investigation tests and outcome
checks, `expected_result` is a literal output marker or
`exit_code=N`. A literal marker confirms the condition only when the command
exits with status 0; use `exit_code=N` for an intentional nonzero result. The
controller infers common markers when a read-only check omits one
(`systemctl is-active` → `active`, `status` → `active (running)`,
`ss`/`netstat` → `LISTEN`, `nmap` → `open`, `curl` → `HTTP/`); a
`purpose=verify` check with no marker and none inferrable is still rejected. A
complete test result that lacks its marker, or exits nonzero when a marker was
expected, counts as a failed or contradicted result.
After two failures or contradictions, the prompt directs the model to revise
its assumptions instead of trying command variants. Model-issued commands run
one shell invocation per tool call. General shell tasks may use a read-only
producer followed by supported output filters; sequential chains, scoped-target
assessments, package-manager steps, and nested shell commands remain standalone.
Before any model-issued state change in a natural-language shell task, a
bounded read-only verification check must determine whether the requested
outcome already exists. The prompt favors direct endpoint/service checks over
broad inventories. Commands typed directly keep their one-command behavior.
Web search and
SearchSploit share a three-call research budget per task. Research cannot come
between a state change and its required read-only outcome check. Explicit action
requests need an outcome check before completion even when no change was needed.
Without one, the agent reports the result as unverified. For a verification
check, `expected_result` is a literal marker expected in captured stdout/stderr
or `exit_code=N`; a literal marker only confirms the condition if the command
exits with status 0. The controller compares it with the recorded result. A
preflight result that is incomplete blocks changes and research until another
bounded read-only check resolves it; the controller allows up to three outcome
checks per state generation. A result that remains unknown is reported as
unverified.
matched preflight ends work when no explicit change was requested. After a
required or explicit change, a matching read-only outcome check ends the tool
loop. The controller verifies the stated condition; it labels the model's
broader assessment `INFERRED` until a goal-specific deterministic validator
exists. The agent prefers
installed tools and requires an explicit installation request before
model-issued package installs. Unknown single-word shell input is sent as
conversation; use `/kali COMMAND` for an arbitrary executable. `/debug` shows
why recent tool calls were rejected and records no-action model replies;
`/evidence [full]` shows recorded command results. A failed or incomplete goal check makes the
controller display recorded evidence instead of a model completion summary.
Explicit create, build, write, update, and restart requests remain open after a
preflight check until a state-changing command runs and receives its own
read-only outcome check.

The executor separates short-lived commands from persistent services.
`run_kali_command` is a one-shot command in a fresh shell; its optional `cwd`
applies only to that invocation. Long-running noninteractive servers use
`start_background`, which returns an app-owned process handle and PID.
`check_process` and `stop_process` accept only handles recorded for the
configured Kali account; the registry survives app restarts and is ignored if
the configured host, port, or user changes. Stopping sends SIGTERM after
checking PID identity. Service output is discarded to avoid unbounded logs. A
live process does not prove its app is healthy, so the workflow must still
verify the requested endpoint. TTY applications that need ongoing input use
`start_interactive`, `read_interactive`, `send_interactive_input`, and
`interrupt_interactive`. Their handles exist only for the app session; app
shutdown requests Ctrl+C and closes the SSH channel without claiming that the
remote program stopped. They reject shells and sudo and block model input when
a credential prompt is observed.
`/processes` lists controller-known handles and their last observed status; it
does not poll Kali.

For `scan ports`, the agent asks which host to scan. A simple scan with a
specified target starts with the top 100 TCP ports and a 45-second host limit.
You can ask explicitly for a full scan. Typing `sudo` alone securely checks the
Kali account's credentials at a local password prompt. On success, it enables
session sudo mode: subsequent commands run through `sudo -S` until `/user`,
`/clear`, or exit. The password stays only in controller memory for that session;
it is not sent to the model or written to disk, and is cleared when sudo mode
ends, authentication fails, or the app exits. A rejected password gets up to
three local retries; the failure is reported without sudo's misleading
"no password was provided" follow-up.

Short `connect kali` requests (including common `cali`/`kalu` typos) run a direct
SSH-backed hostname/user check and do not depend on K2 choosing a command.
Natural-language package installs also use controller-owned stages: inspect
installed and candidate versions, refresh apt metadata once if needed, install
only when a candidate exists, and verify with `dpkg-query`. If a package remains
unavailable, the agent stops before trying `apt-get install` again. Google Chrome
uses Google's official apt repository with a fingerprint-checked keyring and a
`signed-by` source entry; `apt-key` and model-generated download commands are not
used. Temporary sudo access opened by an install request is cleared when that
workflow ends.

## K2 Horizon on Windows

For the quickest launch, run `launch-k2.cmd` from this folder. It starts the
existing local K2 server if needed and then opens the chat program. The server
uses the model file already configured in `start-k2-server.ps1`.

To start the chat manually, leave `llama-server` running and use PowerShell:

```powershell
# Run these commands from the project directory.
$env:DEEP_AGENT_BACKEND = "llama"
$env:DEEP_AGENT_BASE_URL = "http://127.0.0.1:8080/v1"
$env:DEEP_AGENT_MODEL = "k2-horizon"
uv run --with-requirements requirements.txt python deep_agent.py
```

If the server is stopped, start it in a separate PowerShell window first:

```powershell
powershell -ExecutionPolicy Bypass -File .\start-k2-server.ps1 -ContextSize 102400
```

The launcher uses a 100K context (102,400 tokens) with the KV cache in system
RAM so it fits the 12 GB GPU setup. If K2 is already running at another
context size, stop that server before launching again. Adding `-GpuKvCache`
without an explicit context still selects the older 32K GPU cache mode. K2
answers stream into the chat window; its private reasoning content stays hidden.
K2 uses medium reasoning effort by default. Set `DEEP_AGENT_K2_REASONING` to
`low`, `medium`, or `high` for a session-wide choice, or include a phrase such
as "think harder" or "high effort" in a request to run just that turn at high
reasoning.
The prompt shows context used by the last model request against the running
server's context limit. Type `/context` to show it again. K2 gets the limit
from llama-server `/props` and usage from its response; Ollama uses its active
model context and response counters. The count includes the last request's
prompt and reply, so new tool output or messages typed afterward are included
only after the next model request. If the backend does not provide usage or a
context limit, the prompt says that the value is unavailable.

## Ridge 3.7 through Ollama

Keep Ollama running and use the installed Ridge alias:

```powershell
# Run these commands from the project directory.
$env:DEEP_AGENT_BACKEND = "ollama"
$env:DEEP_AGENT_MODEL = "ridge:3.7"
$env:DEEP_AGENT_OLLAMA_NUM_CTX = "32768"
$env:DEEP_AGENT_OLLAMA_THINK = "true"
uv run --with-requirements requirements.txt python deep_agent.py
```

For llama.cpp servers requiring a key, set `DEEP_AGENT_API_KEY` before launch.
For a local server without authentication, it defaults to `local`.

## Conversation context and persistent notes

The terminal agent can now retrieve older captured output with
`list_tool_results(query, offset, limit)` and
`read_tool_result(result_id, stream, offset, limit)`. Retrieval is local and
does not rerun a command. Pages use character offsets, with at most 4,096
characters per read. A page reports `capture_truncated` when the original
transport capture was incomplete; retrieval cannot recover uncaptured data.
Older excerpts retain both heads and tails, execution status, and controller
facts. The complete captured result remains in the local archive.

Model request history is bounded by bytes and message count. Stored history
is trimmed between user turns, preserving controller bookkeeping during a
running task. System instructions, the latest user goal and recent complete
tool exchanges are retained. Omitted user/assistant messages are archived
as `conversation` records and are searchable through `list_tool_results`.
The omission notice is explicit and does not invent a summary. An individual
request that cannot fit is rejected before submission rather than silently
cutting its text.

Defaults are 180,000 serialized UTF-8 bytes and 160 messages, scaled down
when the server context limit is known and accounting for tool schema size.
Override these with `DEEP_AGENT_HISTORY_MAX_BYTES` (minimum 4,096) and
`DEEP_AGENT_HISTORY_MAX_MESSAGES` (minimum 8). The byte scaling estimates
context use; it is not exact tokenization or a server capacity guarantee.
Trimming and memory changes may invalidate prompt-cache reuse at the changed
prefix. The archive is lossless for captured data, but answer quality still
depends on the model retrieving the relevant omitted details.

Session archives are local SQLite files in `runtime/context/`. Each terminal
session and shell `/clear` starts a fresh archive namespace. Old files remain
on disk for inspection; the active model does not automatically reopen them.
These archives and the evidence ledger are separate: the ledger still owns
command truth, verification and controller facts.

Persistent notes now use `lab_notes.sqlite3`, with the existing `lab_notes.md`
imported once without rewriting it. New notes also append a dated markdown
audit line. After import, SQLite owns note state; editing the old markdown
file does not update structured memory. The database is local and ignored
by Git.

- `save_lab_note(text, key, replaces, ttl_days)` records an observation. Use
  stable keys such as `host.service_address` for facts that can change.
- Different values saved under the same key become **disputed**. Free-text
  notes and legacy lines cannot be compared semantically; key them explicitly.
- `replaces=<note_id>` keeps the old version as **superseded**. If other
  competing notes remain, the replacement stays disputed.
- `update_lab_note(note_id, status="stale")` excludes an outdated note from
  the active snapshot. Expiry also makes transient notes stale.
- `read_lab_notes(query, offset, limit, include_inactive)` retrieves IDs,
  statuses and historical versions. Search supports note IDs and keys.
- Reactivation does not verify a statement. Superseded or expired notes need
  a new observation; competing active values must be reconciled explicitly.

The bounded memory snapshot refreshes on subsequent shell model requests,
so updates take effect without restarting. A corrupt database produces an
unavailable-memory notice; it does not reactivate historical markdown claims.

## Controller modules

| Module | Responsibility |
| --- | --- |
| `deep_agent.py` | Task engines, workflow integration and result grounding |
| `agent_tools.py` | Terminal tool schemas and local argument validation |
| `model_client.py` | HTTP transport with explicit settings and callbacks |
| `model_protocol.py` | Tool-call decoding and Ollama history adaptation |
| `model_streaming.py` | Generation display and SSE/NDJSON assembly |
| `conversation_context.py` | Result archive, pagination, compaction and history bounds |
| `persistent_memory.py` | Versioned notes, conflict handling and expiry |
| `terminal_cli.py` | Terminal session routing through the controller API |

These modules belong to the terminal agent. Browser controls continue to use
the browser implementation. No new Python dependencies are required.

## Offline regression tests

Run the active chat and controller tests with:

```powershell
uv run --with-requirements requirements.txt --with pytest python -m pytest -q test_kali_chat.py test_unrestricted_agent.py test_network_preflight.py test_context_memory.py
```
