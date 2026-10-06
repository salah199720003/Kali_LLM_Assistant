# Kali network preflight case study

## Observed baseline

The attached `agent-27b` session contained 35 model replies. During the attack
task, the agent later recorded Kali's `eth0` address as `10.0.2.15` and its
default gateway as `10.0.2.2`. It had already tried callback addresses such as
`192.168.56.101`; Metasploit reported that it could not bind that handler. The
agent then spent more model rounds on unavailable modules and hand-written SMB
negotiate packets. Its subsequent MS17-010 scanner result was a lead, while its
exploit attempts produced no session.

## Harness change

For an XXS remote attack request with an explicit target IP, the harness now
records `ip -j -4 address show` and `ip -j -4 route get TARGET` before asking the
model for a command. The evidence is appended as controller-collected data and
leaves the unrestricted command path available. Before the first exploit or
payload tool call, the model must attach a structured `network_plan`. The
controller checks the target, route source, listener bind address, LHOST, and
any verified callback evidence. `unverified` is an explicit valid status; the
controller does not turn a route into proof of return reachability. A handler
bind error invalidates the plan, so another attempt needs a refreshed network
assessment and plan.

## Regression replay

`test_network_preflight.py` replays the recorded topology using mocked SSH: the
target is `192.168.56.102`, Kali's address is `10.0.2.15`, and the route uses
gateway `10.0.2.2`. It checks that a proposed default listener at
`192.168.56.101` is held before submission, while a corrected local LHOST can
proceed with callback reachability explicitly marked unverified. The replay
also checks a clean first-round plan. It performs no connection to Kali or the
Windows VM.

The plan check covers literal Metasploit settings and common hand-built SMB
packet commands. Arbitrary obfuscated scripts cannot be classified perfectly;
the model prompt still requires a network plan before its first exploit call.
