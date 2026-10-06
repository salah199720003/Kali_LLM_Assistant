"""Drive the real K2 agent (launch-k2 path) with prompt-marker-fed input.
Feeds the next line when the agent actually asks for it: typed prompts
("You [shell]>"), local getpass prompts (SSH/sudo passwords), so the timing
matches a human at the keyboard. Session transcript goes to k2_session.log.
"""
import os
import subprocess
import sys
import threading
import time
STEPS = [
    ("You [chat]>", "$shell"),
    ("You [shell]>", "connect kali"),
    ("You [shell]>", "sudo"),
    ("You [shell]>", "exploit 127.0.0.1"),
    ("You [shell]>", "what were the results?"),
    ("You [shell]>", "exit"),
]
env = dict(os.environ)
env.update({
    "DEEP_AGENT_BACKEND": os.environ.get("DRIVE_BACKEND", "llama"),
    "DEEP_AGENT_BASE_URL": os.environ.get("DRIVE_BASE_URL", "http://127.0.0.1:8080/v1"),
    "DEEP_AGENT_MODEL": os.environ.get("DRIVE_MODEL", "k2-horizon"),
    "DEEP_AGENT_VM_PASSWORD": "kali",
    "PYTHONUNBUFFERED": "1",
})
proc = subprocess.Popen(
    ["uv", "run", "--with-requirements", "requirements.txt", "python", "deep_agent.py"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
)
pending = list(STEPS)
step_deadline = time.time() + int(os.environ.get("DRIVE_STEP_TIMEOUT", "150"))
hard_deadline = time.time() + int(os.environ.get("DRIVE_HARD_TIMEOUT", "1500"))
watchdog = threading.Event()
def feed_when_asked():
    while pending and time.time() < hard_deadline:
        marker, line = pending[0]
        if watchdog.wait(timeout=2):
            return
        time.sleep(0.2)
watchdog_thread = threading.Thread(target=feed_when_asked, daemon=True)
watchdog_thread.start()
log = open(os.environ.get("DRIVE_LOG", "k2_session.log"), "w", encoding="utf-8", errors="replace")
buffer = ""
try:
    while time.time() < hard_deadline:
        # Prompts ("You [shell]>", "Kali password:") print no newline, so read
        # raw chunks instead of lines or the feeder never sees them.
        chunk = os.read(proc.stdout.fileno(), 4096)
        if not chunk:
            if proc.poll() is not None:
                break
            time.sleep(0.1)
            continue
        chunk = chunk.decode("utf-8", "replace")
        buffer += chunk
        log.write(chunk)
        log.flush()
        if pending:
            marker, line = pending[0]
            if marker in buffer:
                buffer = ""
                try:
                    proc.stdin.write(line + "\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    break
                pending.pop(0)
                step_deadline = time.time() + int(os.environ.get("DRIVE_STEP_TIMEOUT", "150"))
        if time.time() > step_deadline and pending:
            # feed anyway to avoid wedging on a marker that never prints
            marker, line = pending.pop(0)
            try:
                proc.stdin.write(line + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                break
            step_deadline = time.time() + int(os.environ.get("DRIVE_STEP_TIMEOUT", "150"))
finally:
    try:
        proc.kill()
    except OSError:
        pass
    log.close()
print("driver finished; transcript: k2_session.log")
