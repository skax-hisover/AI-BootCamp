from src.utils.memory import SessionMemory


def test_session_memory_limit() -> None:
    memory = SessionMemory()
    for i in range(10):
        memory.add("s1", "user", f"msg-{i}")

    recent = memory.get("s1", limit=3)
    assert len(recent) == 3
    assert recent[0]["content"] == "msg-7"
    assert recent[-1]["content"] == "msg-9"
