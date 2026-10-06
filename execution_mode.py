"""Execution mode shared by the terminal controller and its SSH transport."""

import os


def unrestricted_execution_enabled(model: str | None = None) -> bool:
    selected_model = model if model is not None else os.environ.get("DEEP_AGENT_MODEL", "")
    return (selected_model.strip().lower() in {"agent-27b", "qwen3.8-flash-next-coder-iq1_m"}
            and os.environ.get("DEEP_AGENT_EXECUTION_MODE", "guarded").strip().lower() == "unrestricted")
