"""
tests/devswarm/test_repo_map.py — Unit tests for repo map AST signature extraction and size bounding.
"""

from pathlib import Path
from devswarm.tools.repo_map import generate_repo_map, extract_python_signatures


def test_extract_python_signatures(tmp_path):
    py_file = tmp_path / "module.py"
    py_file.write_text("""
class Calculator:
    def add(self, a, b):
        return a + b

    async def fetch(self, url):
        pass

def top_level_helper(x, y):
    return x * y
""", encoding="utf-8")

    sigs = extract_python_signatures(py_file)
    assert any("class Calculator" in s for s in sigs)
    assert any("def add(self, a, b)" in s for s in sigs)
    assert any("def fetch(self, url)" in s for s in sigs)
    assert any("def top_level_helper(x, y)" in s for s in sigs)


def test_repo_map_generation_and_size_capping(tmp_path):
    # Create multiple files
    src = tmp_path / "src"
    src.mkdir()
    (src / "service.py").write_text("class UserService:\n    def get_user(self, user_id):\n        pass\n")
    (src / "models.py").write_text("class User:\n    pass\n")

    # Ignored directory
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("should_be_ignored")

    repo_map = generate_repo_map(tmp_path, max_chars=1000)
    assert "<UNTRUSTED_REPO_MAP>" in repo_map
    assert "src/service.py" in repo_map
    assert "class UserService" in repo_map
    assert ".git" not in repo_map

    # Test strict size truncation
    tight_map = generate_repo_map(tmp_path, max_chars=100)
    assert len(tight_map) <= 150
    assert "[TRUNCATED" in tight_map

