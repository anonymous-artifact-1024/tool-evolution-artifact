"""Agent-visible messages built only from the public task projection."""

from __future__ import annotations

from saner_exp.data.task import AgentTask


SYSTEM_PROMPT = """You are a Linux kernel fault-localization agent. Investigate the given bug report using the available tools. Emit at most one tool call in each turn. The repository is read-only. When ready, call the result-submission tool with a ranked list of at most ten repository-relative source-file paths. A plain-text answer does not finish the task."""


def build_initial_messages(task: AgentTask) -> tuple[dict, dict]:
    user = f"Kernel version: {task.kernel_version}\nTitle: {task.title}\n\nDescription:\n{task.description}"
    return ({"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user})
