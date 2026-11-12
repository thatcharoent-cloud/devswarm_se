"""
tests/devswarm/test_manifest_memory.py — Unit tests for RunManifest, ArtifactStore, and ProjectMemory.
"""

import json
from pathlib import Path

from devswarm.platform.manifest import build_manifest, RunManifest
from devswarm.platform.artifacts import ArtifactStore
from devswarm.platform.memory import ProjectMemory


def test_manifest_generation(tmp_path):
    manifest = build_manifest(
        run_id="run-test-01",
        goal="Test feature build",
        repo_path=tmp_path,
        model="qwen2.5-coder:7b",
        num_ctx=16384,
        config_snapshot={"api_key": "sk-secret12345", "concurrency": 1},
        prompt_dict={"orchestrator": "You are orchestrator."},
    )

    assert manifest.run_id == "run-test-01"
    # Ensure secret is redacted from config snapshot
    assert manifest.config_snapshot["api_key"] == "[REDACTED_SECRET]"
    assert manifest.num_ctx == 16384
    assert "orchestrator" in manifest.prompt_hashes

    save_path = manifest.save(tmp_path)
    assert save_path.exists()
    saved_data = json.loads(save_path.read_text(encoding="utf-8"))
    assert saved_data["run_id"] == "run-test-01"
    assert saved_data["config_snapshot"]["api_key"] == "[REDACTED_SECRET]"


def test_artifact_store(tmp_path):
    store = ArtifactStore(run_dir=tmp_path)

    p1 = store.write_artifact("report.md", "# Test Report\nEverything passed.")
    assert p1.exists()

    p2 = store.write_artifact("diff.patch", "--- a/file\n+++ b/file\n", content_type="text/x-diff")
    assert p2.exists()

    items = store.list_artifacts()
    assert len(items) == 2
    names = [i["name"] for i in items]
    assert "report.md" in names
    assert "diff.patch" in names


def test_project_memory(tmp_path):
    mem_file = tmp_path / "memory.jsonl"
    memory = ProjectMemory(memory_file=mem_file)

    memory.record("architecture", "db", "Use SQLite for queue storage", source_task="task-1")
    memory.record("convention", "formatting", "Black + ruff for python formatting", source_task="task-2")

    entries = memory.read_all()
    assert len(entries) == 2

    summary = memory.get_summary(max_chars=2000)
    assert "<UNTRUSTED_PROJECT_MEMORY>" in summary
    assert "SQLite for queue storage" in summary
    assert "Black + ruff" in summary
    assert "</UNTRUSTED_PROJECT_MEMORY>" in summary

