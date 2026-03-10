from pathlib import Path

from scripts.migrate_cache import migrate_cache


def test_migrate_cache_legacy_to_final(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    legacy = index_dir / "graph_state_cache.json"
    legacy.write_text('{"s1": {"cache_kind": "legacy"}}', encoding="utf-8")

    ok, message = migrate_cache(index_dir=index_dir, force=False)
    assert ok is True
    assert "migrated" in message
    assert (index_dir / "final_answer_cache.json").exists()
    assert (index_dir / "graph_state_cache.backup.json").exists()


def test_migrate_cache_skips_when_final_exists_without_force(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "graph_state_cache.json").write_text('{"s1": {"legacy": 1}}', encoding="utf-8")
    (index_dir / "final_answer_cache.json").write_text('{"s1": {"new": 1}}', encoding="utf-8")

    ok, message = migrate_cache(index_dir=index_dir, force=False)
    assert ok is False
    assert "already exists" in message
