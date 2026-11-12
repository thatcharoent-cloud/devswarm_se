"""
devswarm.agents.parser — Tolerant tool-call extractor and dirty JSON parser.

Handles quirks of local models (e.g. Qwen 2.5 / 3.5, Mistral):
  1. Standard Ollama `message.tool_calls` structures.
  2. Markdown code fences containing JSON tool calls (```json ... ```).
  3. XML-style `<tool_call>...</tool_call>` wrappers.
  4. Dirty JSON syntax (single quotes, trailing commas, Python booleans/None).
  5. Alternative schema keys ("tool" / "function", "parameters" / "arguments").
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParsedToolCall:
    """Standardized tool invocation extracted from model output."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    tool_call_id: str | None = None


def sanitize_dirty_json(text: str) -> str:
    """Repair common syntax issues found in local LLM generated JSON strings."""
    s = text.strip()

    # Strip markdown code blocks if wrapped
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if fence_match:
        s = fence_match.group(1).strip()

    # Replace Python True/False/None with JSON true/false/null
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)

    # Remove trailing commas before closing braces or brackets
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # Replace single quotes with double quotes if not containing valid double quotes
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')
    elif "'" in s:
        # Replace key single quotes: 'key': -> "key":
        s = re.sub(r"'([a-zA-Z0-9_]+)'\s*:", r'"\1":', s)
        # Replace string literal single quotes when not escaped
        s = re.sub(r":\s*'([^']*)'", r': "\1"', s)

    return s


def repair_truncated_json(text: str) -> str:
    """Repair truncated JSON where strings, arrays, or objects were cut off mid-stream."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Find the first opening brace or bracket
    start_idx = -1
    for i, c in enumerate(cleaned):
        if c in ("{", "["):
            start_idx = i
            break
    if start_idx == -1:
        return cleaned

    s = cleaned[start_idx:]
    in_string = False
    escape = False
    stack: list[str] = []

    for c in s:
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if not in_string:
            if c in ("{", "["):
                stack.append(c)
            elif c == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif c == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

    repaired = s
    if in_string:
        repaired += '"'

    # Strip trailing commas, colons, or whitespace before closing delimiters
    repaired = re.sub(r"[,:\s]+$", "", repaired)

    # Close any unclosed braces/brackets in LIFO order
    for delim in reversed(stack):
        if delim == "{":
            repaired += "}"
        elif delim == "[":
            repaired += "]"

    return repaired


def parse_single_dict(d: dict[str, Any]) -> ParsedToolCall | None:
    """Extract a tool call from a dictionary representing function or tool."""
    if not isinstance(d, dict):
        return None

    # Pattern A: Standard Ollama / OpenAI: {"function": {"name": ..., "arguments": ...}}
    if "function" in d and isinstance(d["function"], dict):
        fn = d["function"]
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(sanitize_dirty_json(args))
            except Exception:
                args = {}
        if name:
            return ParsedToolCall(name=name, arguments=args if isinstance(args, dict) else {}, tool_call_id=d.get("id"))

    # Pattern B: Direct format: {"name": "read_file", "arguments": {...}}
    name = d.get("name") or d.get("tool") or d.get("action")
    args = d.get("arguments") or d.get("parameters") or d.get("args") or d.get("input")

    if name and isinstance(name, str):
        if isinstance(args, str):
            try:
                args = json.loads(sanitize_dirty_json(args))
            except Exception:
                args = {}
        return ParsedToolCall(
            name=name.strip(),
            arguments=args if isinstance(args, dict) else {},
            raw_text=json.dumps(d),
            tool_call_id=d.get("id"),
        )

    return None


def extract_tool_calls(
    response_tool_calls: list[dict[str, Any]] | None = None,
    content: str | None = None,
) -> list[ParsedToolCall]:
    """Extract tool calls tolerantly from standard response tool_calls or response content."""
    results: list[ParsedToolCall] = []

    # 1. Check native Ollama tool calls
    if response_tool_calls:
        for tc in response_tool_calls:
            parsed = parse_single_dict(tc)
            if parsed:
                results.append(parsed)

    if results:
        return results

    if not content or not content.strip():
        return []

    text = content.strip()

    # 2. Check XML-like <tool_call> tags
    xml_matches = re.findall(r"<tool_call>([\s\S]*?)</tool_call>", text)
    if xml_matches:
        for m in xml_matches:
            cleaned = sanitize_dirty_json(m)
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict):
                    p = parse_single_dict(data)
                    if p:
                        p.raw_text = m
                        results.append(p)
                elif isinstance(data, list):
                    for item in data:
                        p = parse_single_dict(item)
                        if p:
                            results.append(p)
            except Exception:
                pass

    if results:
        return results

    # 3. Check JSON code blocks: ```json ... ```
    fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    for fm in fence_matches:
        cleaned = sanitize_dirty_json(fm)
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                # Check if it has a list of tool_calls
                if "tool_calls" in data and isinstance(data["tool_calls"], list):
                    for item in data["tool_calls"]:
                        p = parse_single_dict(item)
                        if p:
                            results.append(p)
                else:
                    p = parse_single_dict(data)
                    if p:
                        p.raw_text = fm
                        results.append(p)
            elif isinstance(data, list):
                for item in data:
                    p = parse_single_dict(item)
                    if p:
                        results.append(p)
        except Exception:
            pass

    if results:
        return results

    # 4. Check if text itself is direct JSON
    try:
        data = json.loads(sanitize_dirty_json(text))
        if isinstance(data, dict):
            p = parse_single_dict(data)
            if p:
                p.raw_text = text
                return [p]
        elif isinstance(data, list):
            direct_results = []
            for item in data:
                p = parse_single_dict(item)
                if p:
                    direct_results.append(p)
            if direct_results:
                return direct_results
    except Exception:
        pass

    # 5. Extract balanced JSON object within surrounding text
    start_idx = -1
    for i, ch in enumerate(text):
        if ch == "{":
            start_idx = i
            break

    if start_idx != -1:
        depth = 0
        end_idx = -1
        in_string = False
        escape = False
        for i in range(start_idx, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break
        if end_idx != -1:
            raw_json = text[start_idx:end_idx]
            try:
                data = json.loads(sanitize_dirty_json(raw_json))
                if isinstance(data, dict):
                    p = parse_single_dict(data)
                    if p:
                        p.raw_text = raw_json
                        results.append(p)
            except Exception:
                pass

    return results
