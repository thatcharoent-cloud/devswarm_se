"""
devswarm.pr — Pull Request generator creating comprehensive PR.md reports.

Generates .swarm/runs/<run_id>/PR.md with:
  - Goal and executive summary
  - Completed task DAG breakdown
  - Deterministic Quality Gate verification matrix
  - Automated Reviewer verdict and security comments
  - Time and token budget telemetry ($0.00 pricing, throughput, TTFT)
  - Git diff statistics
  - Explicit CLI commands for human approval and merge
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from devswarm.gates import GateSuiteResult
from devswarm.reviewer import ReviewReport

logger = logging.getLogger(__name__)


class PRGenerator:
    """Generates standardized Markdown Pull Request documentation."""

    @classmethod
    def generate(
        cls,
        run_id: str,
        goal: str,
        repo_path: Path | str,
        base_commit: str,
        integration_branch: str,
        tasks: list[dict[str, Any]],
        gate_result: GateSuiteResult,
        review_report: ReviewReport,
        budget_summary: dict[str, Any],
        output_dir: Path | str,
        traceability_markdown: str = "",
        role_telemetry: list[dict[str, Any]] | None = None,
        sources_cited: list[dict[str, str]] | None = None,
        cloud_accounting: dict[str, Any] | None = None,
        unresolved_issues: list[str] | None = None,
    ) -> Path:
        """Create and write comprehensive hybrid PR.md artifact."""
        out_path = Path(output_dir) / "PR.md"
        repo_dir = Path(repo_path).resolve()

        # Extract git diff stats
        diff_stats = ""
        try:
            res = subprocess.run(
                ["git", "diff", "--stat", f"{base_commit}..{integration_branch}"],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            diff_stats = res.stdout.strip()
        except Exception:
            diff_stats = "Git diff stats unavailable."

        # Format Tasks Table
        task_rows = []
        for t in tasks:
            tid = t.get("id", "")
            role = t.get("role", "")
            title = t.get("title", "")
            state = t.get("state", "done")
            scope = ", ".join(t.get("write_scope", [])) or "none"
            task_rows.append(f"| `{tid}` | `{role}` | `{state}` | `{scope}` | {title} |")
        tasks_table = (
            "| Task ID | Role | State | Write Scope | Title |\n"
            "|---|---|:---:|---|---|\n"
            + "\n".join(task_rows)
        )

        # Format Quality Gates Table
        gate_rows = []
        for g_name, g_res in gate_result.results.items():
            status_icon = "✅ PASSED" if g_res.passed else "❌ FAILED"
            req_str = "Yes" if g_res.required else "No"
            gate_rows.append(f"| `{g_name}` | {status_icon} | {req_str} | {g_res.duration_s:.2f}s |")
        gates_table = (
            "| Gate | Result | Required | Duration |\n"
            "|---|:---:|:---:|:---:|\n"
            + "\n".join(gate_rows)
        )

        # Reviewer Strengths & Concerns
        strengths_list = "\n".join(f"- {s}" for s in review_report.strengths) if review_report.strengths else "- Code meets quality expectations."
        concerns_list = "\n".join(f"- {c}" for c in review_report.concerns) if review_report.concerns else "- None noted."

        # High-Risk Change Alert
        high_risk_notice = ""
        if gate_result.has_high_risk_changes:
            high_risk_notice = (
                "\n> [!CAUTION]\n"
                "> **HIGH-RISK PATHS MODIFIED**: This pull request modifies security-sensitive files:\n"
                + "\n".join(f"> - `{f}`" for f in gate_result.high_risk_files)
                + "\n> **Human approval is mandatory before merge.**\n"
            )

        status_badge = "READY FOR MERGE" if (review_report.verdict == "APPROVE" and not gate_result.has_high_risk_changes) else "NEEDS HUMAN APPROVAL"

        # Format Traceability Matrix section
        trace_section = ""
        if traceability_markdown:
            trace_section = f"\n---\n\n## 5. Specification & Acceptance Traceability\n{traceability_markdown}\n"

        # Format Roles & Cost Telemetry Table
        roles_table = ""
        if role_telemetry:
            r_rows = []
            for r in role_telemetry:
                cost_s = "$0.00 (Local)" if r.get("cost_usd", 0.0) == 0.0 else f"${r.get('cost_usd', 0.0):.4f}"
                r_rows.append(
                    f"| `{r.get('role')}` | `{r.get('provider')}` | `{r.get('model')}` | "
                    f"{r.get('input_tokens', 0):,} | {r.get('output_tokens', 0):,} | {cost_s} |"
                )
            roles_table = (
                "\n### Specialist Roles & Cost Accounting\n"
                "| Role | Provider | Model | Input Tokens | Output Tokens | Cost (USD) |\n"
                "|---|---|---|:---:|:---:|:---:|\n"
                + "\n".join(r_rows)
                + "\n"
            )

        # Format Sources Cited section
        sources_section = ""
        if sources_cited:
            s_rows = []
            for s in sources_cited:
                st = s.get("status", "VERIFIED")
                icon = "✅ `[VERIFIED]`" if "VERIFIED" in st else "⚠️ `[UNVERIFIED]`"
                s_rows.append(f"| {s.get('title', 'Doc')} | `{s.get('url', '')}` | {icon} |")
            sources_section = (
                "\n---\n\n## 6. Research Sources & Citations\n"
                "| Title | URL | Verification |\n"
                "|---|---|:---:|\n"
                + "\n".join(s_rows)
                + "\n"
            )

        # Format Outbound Cloud Security Accounting
        cloud_sec_section = ""
        if cloud_accounting:
            total_spent = cloud_accounting.get("spent_usd_total", 0.0)
            classes = ", ".join(f"`{c}`" for c in cloud_accounting.get("content_classes", ["goal", "spec", "architecture"]))
            bytes_out = cloud_accounting.get("bytes_sent", 0)
            cloud_sec_section = (
                f"\n### Cloud Data Policy & Outbound Telemetry\n"
                f"- **Total Cloud Spend:** **${total_spent:.4f} USD**\n"
                f"- **Outbound Content Classes:** {classes}\n"
                f"- **Total Bytes Sent to Cloud:** {bytes_out:,} bytes\n"
            )

        # Unresolved issues
        unresolved_section = ""
        if unresolved_issues:
            unresolved_section = "\n### Unresolved Issues & Open Questions\n" + "\n".join(f"- ❓ {q}" for q in unresolved_issues) + "\n"

        markdown = f"""# Pull Request: {goal}

**Status:** `{status_badge}`  
**Run ID:** `{run_id}`  
**Integration Branch:** `{integration_branch}`  
**Base Commit:** `{base_commit[:8] if len(base_commit) >= 8 else base_commit}`
{high_risk_notice}
---

## 1. Executive Summary
{review_report.summary}

---

## 2. Completed Tasks
{tasks_table}

---

## 3. Deterministic Quality Gates
{gates_table}

**Gate Suite Summary:** {gate_result.summary}

---

## 4. Senior Reviewer Evaluation
- **Verdict:** `{review_report.verdict}` (Confidence: {review_report.confidence_score * 100:.0f}%)
- **Security Assessment:** {review_report.security_assessment}

### Strengths
{strengths_list}

### Considerations & Concerns
{concerns_list}
{unresolved_section}{trace_section}
---

## Resource & Telemetry Accounting
- **Local Model Cost:** **$0.000000** *(100% Local Inference via Ollama on Apple Silicon Metal GPU)*
- **Total Runtime:** {budget_summary.get('elapsed_minutes', 0.0):.2f} minutes
- **Total LLM Inferences:** {budget_summary.get('total_llm_calls', 0)}
- **Tokens In / Out:** {budget_summary.get('total_input_tokens', 0):,} / {budget_summary.get('total_output_tokens', 0):,}
{roles_table}{cloud_sec_section}{sources_section}
---

## Git Diff Statistics
```text
{diff_stats}
```

---

## Human Product Owner Actions

To merge this pull request into the default branch:
```bash
python main.py approve --run-id {run_id}
```

To reject this pull request and retain branch for inspection:
```bash
python main.py reject --run-id {run_id} --reason "Your feedback here"
```
"""

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        logger.info("Generated PR artifact at %s", out_path)
        return out_path

