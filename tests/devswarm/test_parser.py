"""
tests/devswarm/test_parser.py — Tests for tolerant tool-call parsing and dirty JSON repair.
"""

from devswarm.agents.parser import extract_tool_calls, sanitize_dirty_json


def test_standard_ollama_tool_calls() -> None:
    raw_calls = [
        {
            "function": {
                "name": "read_file",
                "arguments": {"path": "calc.py", "start_line": 1, "end_line": 10},
            }
        }
    ]
    parsed = extract_tool_calls(response_tool_calls=raw_calls)
    assert len(parsed) == 1
    assert parsed[0].name == "read_file"
    assert parsed[0].arguments == {"path": "calc.py", "start_line": 1, "end_line": 10}


def test_markdown_json_block_tool_call() -> None:
    content = """
Here is my plan. I will read the file:
```json
{
  "name": "read_file",
  "arguments": {
    "path": "calc.py"
  }
}
```
Please execute this tool.
"""
    parsed = extract_tool_calls(content=content)
    assert len(parsed) == 1
    assert parsed[0].name == "read_file"
    assert parsed[0].arguments == {"path": "calc.py"}


def test_xml_tool_call_tags() -> None:
    content = """
<tool_call>
{"name": "finish_task", "arguments": {"summary": "Implementation completed"}}
</tool_call>
"""
    parsed = extract_tool_calls(content=content)
    assert len(parsed) == 1
    assert parsed[0].name == "finish_task"
    assert parsed[0].arguments["summary"] == "Implementation completed"


def test_dirty_json_single_quotes_and_trailing_commas() -> None:
    content = """
```json
{
  'name': 'edit_file',
  'arguments': {
    'path': 'calc.py',
    'old_str': 'return a + b',
    'new_str': 'return float(a + b)',
  },
}
```
"""
    parsed = extract_tool_calls(content=content)
    assert len(parsed) == 1
    assert parsed[0].name == "edit_file"
    assert parsed[0].arguments["path"] == "calc.py"
    assert parsed[0].arguments["new_str"] == "return float(a + b)"


def test_alternative_keys_tool_and_parameters() -> None:
    content = """
{"tool": "run_command", "parameters": {"command": "pytest tests/ -v"}}
"""
    parsed = extract_tool_calls(content=content)
    assert len(parsed) == 1
    assert parsed[0].name == "run_command"
    assert parsed[0].arguments == {"command": "pytest tests/ -v"}


def test_plain_text_no_tool_calls() -> None:
    content = "I have reviewed the code and everything looks good."
    parsed = extract_tool_calls(content=content)
    assert len(parsed) == 0


def test_sanitize_dirty_json_python_booleans() -> None:
    raw = "{'active': True, 'verified': False, 'details': None,}"
    cleaned = sanitize_dirty_json(raw)
    assert "true" in cleaned
    assert "false" in cleaned
    assert "null" in cleaned
    assert not cleaned.endswith(",}")


def test_repair_truncated_json_unterminated_string() -> None:
    from devswarm.agents.parser import repair_truncated_json
    import json

    truncated = '{"test_files": [{"path": "tests/test.py", "content": "import pytest\\ndef test():\\n    assert True'
    repaired = repair_truncated_json(truncated)
    data = json.loads(repaired, strict=False)
    assert "test_files" in data
    assert data["test_files"][0]["path"] == "tests/test.py"
    assert "assert True" in data["test_files"][0]["content"]


def test_qa_test_suite_resilience_and_ac_mapping() -> None:
    from devswarm.agents.qa_schema import QATestSuite

    # Test auto-synthesis of ac_mapping when missing
    data = {
        "test_files": [
            {
                "path": "tests/test_calculator.py",
                "content": "def test_ac_1_add(): pass\ndef test_ac_2_sub(): pass\n",
            }
        ]
    }
    suite = QATestSuite.model_validate(data)
    assert "AC-1" in suite.ac_mapping
    assert "AC-2" in suite.ac_mapping
    assert suite.ac_mapping["AC-1"] == ["test_ac_1_add"]
    assert suite.ac_mapping["AC-2"] == ["test_ac_2_sub"]

    # Test alternative top-level key "tests"
    alt_data = {
        "tests": [
            {
                "path": "tests/test_foo.py",
                "content": "def test_foo(): pass\n",
            }
        ]
    }
    alt_suite = QATestSuite.model_validate(alt_data)
    assert len(alt_suite.test_files) == 1
    assert "AC-1" in alt_suite.ac_mapping


