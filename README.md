# Local Qwen3.8, Bonsai, K2 and DeepHat agent with Kali shell mode

## Current launch profile: Qwen3.8 27B

Run `launch-agent.cmd` or the existing `launch-k2.cmd` shortcut to use the
already-installed `qwen3.8:27b` model through Ollama. The launcher checks the
local model before opening the agent; it does not download model files. The
profile requests xhigh reasoning and a 32,768-token context from Ollama.

Use `launch-bonsai.cmd` to keep running Bonsai 2 27B with its separate
llama.cpp server and 102,400-token context.

```bash
./launch-agent.cmd
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

K2 remains available through `launch-k2.ps1` for comparisons after stopping
Bonsai. That profile still uses 102,400 context. `launch-k2.cmd` follows the
default Qwen profile for compatibility with the existing desktop shortcut.

DeepHat remains available as a separate profile. Run `launch-deephat.cmd` to
start its server and agent at 102,400 context. The model file is read from the
existing Ollama cache; no second download is needed.

`deep_agent.py` keeps conversation history in the running session. The default
launcher selects Qwen3.8 27B through the existing Ollama backend; the explicit
Bonsai and K2 launchers remain available. It starts
in `You [chat]>` for ordinary conversation. Type `/shell` to switch to
`You [shell]>` for Kali commands and lab actions, and `/chat` to return. Both
modes keep their own history. `exit` quits; `/clear` resets the current mode.

Shell mode shows each Kali command, live output, elapsed time, and exit status
in the same window. The legacy terminal snapshot and local assessment reports
remain in the development workspace and are excluded from source control.

The editable prompts are [`chat_system_prompt.txt`](chat_system_prompt.txt)
and [`kali_system_prompt.txt`](kali_system_prompt.txt). Restart the agent after
editing either prompt.

## Kali connection

The agent connects when a command is first needed. The defaults match the
existing VirtualBox SSH forwarding setup: `kali@127.0.0.1:2222`. Connect once
from Windows to trust Kali's SSH host key:

```powershell
ssh -p 2222 kali@127.0.0.1
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
commands. The controller stops on a repeated command, timeout, interruption,
unknown exit status, authorization failure, or exhausted command budget. A
command typed directly still runs once. Individual command timeouts apply.
Multiline commands, including file-writing heredocs, are supported up to
64,000 characters. Tool-planning replies are held until complete so partial
text cannot appear as an answer before a command runs.
If the model does not issue a usable tool call, no command runs and shell mode
tells you so.
Natural-language requests in shell mode always offer the Kali tool, including
follow-ups such as `Go ahead`. Asking `what were the results?` summarizes
recorded commands and output without running more commands. The same reporting
path is used when a command sequence stops; if the model cannot summarize, the recorded
results are displayed directly.

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
The prompt shows context used by the last model request against the running
server's context limit. Type `/context` to show it again. K2 gets the limit
from llama-server `/props` and usage from its response; Ollama uses its active
model context and response counters. The count includes the last request's
prompt and reply, so new tool output or messages typed afterward are included
only after the next model request. If the backend does not provide usage or a
context limit, the prompt says that the value is unavailable.

## Qwen through Ollama

Keep Ollama running and set the model name you have installed:

```powershell
# Run these commands from the project directory.
$env:DEEP_AGENT_BACKEND = "ollama"
$env:DEEP_AGENT_MODEL = "qwen3.8:27b"
$env:DEEP_AGENT_OLLAMA_NUM_CTX = "32768"
$env:DEEP_AGENT_OLLAMA_THINK = "xhigh"
uv run --with-requirements requirements.txt python deep_agent.py
```

For llama.cpp servers requiring a key, set `DEEP_AGENT_API_KEY` before launch.
For a local server without authentication, it defaults to `local`.

## Tests

Run the active chat and controller tests with:

```powershell
uv run --with-requirements requirements.txt --with pytest python -m pytest -q test_kali_chat.py
```
