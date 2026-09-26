from saner_exp.agent import build_initial_messages
from saner_exp.data.task import AgentTask


def test_prompt_contains_only_agent_task_fields_needed_by_protocol():
    task = AgentTask("SECRET-ID-DO-NOT-SHOW", "Crash title", "Crash description", "3.3.3")
    messages = build_initial_messages(task)
    rendered = "\n".join(message["content"] for message in messages)
    assert "Crash title" in rendered
    assert "Crash description" in rendered
    assert "3.3.3" in rendered
    assert "SECRET-ID-DO-NOT-SHOW" not in rendered
