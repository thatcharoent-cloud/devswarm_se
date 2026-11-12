# 🚀 DevSwarm — Autonomous Software Engineering Swarm

An autonomous multi-agent software engineering swarm supporting **cross-platform environments (macOS, Linux, and Windows WSL2)**: orchestrates role-specialized agents (Product Manager, Software Architect, QA Test Engineer, Backend/Frontend Coders, and Senior Reviewer) across **Cloud LLMs** (SiliconFlow, DeepSeek, OpenAI, Anthropic) and **Local Models** (Ollama) with isolated Git worktree execution and deterministic quality gates.

---

## 🏛️ Architecture Overview

```
                               Human Product Owner
                                        │
                                        ▼
                            [PM Agent (Cloud/Local)]
                                        │  Generates spec.md with numbered AC-1..n
                                        ▼
                       🛑 HUMAN GATE 1: approve-spec [RUN_ID]
                                        │
                                        ▼
                        [Architect Agent (Cloud/Local)]
                                        │  Generates architecture.md & interface contract stubs
                                        ▼
                      🛑 HUMAN GATE 2: approve-design [RUN_ID]
                         (or --auto-approve-design for low risk)
                                        │
                                        ▼
                           [QA Agent (Cloud/Local)]
                                        │  Test-First Suite (tests/)
                                        │  RED-check verified in sandbox (must fail on NotImplementedError)
                                        ▼
                          Two-Pool Resource Scheduler
                      ┌─────────────────┴─────────────────┐
                      │                                   │
              [Local Pool: Concurrency 1]         [Cloud Pool: Concurrency N]
           Ollama Qwen 9B (Metal GPU)         PM / Architect / QA / Debugger
           Coding agents in git worktrees      Zero GPU memory pressure
                      │                                   │
                      └─────────────────┬─────────────────┘
                                        ▼
                           Deterministic Quality Gates
                         • Bytecode Compilation (py_compile)
                         • Linter (ruff)
                         • Test Suite (pytest)
                         • AST Security & Secret Scanning
                                        │
                     ┌──────────────────┴──────────────────┐
                     │ Gate Failure                        │ Gate Passed
                     ▼                                     ▼
             Self-Healing Loop                    Integration Branch Commit
      • Pure-Python failure diagnostics                     │
      • Up to MAX_FIX_ATTEMPTS local repair                 │
      • Optional: --escalate-to-cloud                       │
                     │                                     │
                     └──────────────────┬──────────────────┘
                                        ▼
                     Senior Reviewer & PR Generator
                      • Traceability Matrix (AC -> Task -> Test -> Status)
                      • Specialist Roles & Dollar Cost Telemetry
                      • Outbound Cloud Data Accounting
                      • Research Citations & Source Verification
                                        │
                                        ▼
                      🛑 HUMAN GATE 3: approve [RUN_ID]
                                        │
                                        ▼
                            Merge into main branch
```

---

## ✨ Key Capabilities

1. **Hybrid Cost & Quality Efficiency**:
   - **Product Manager**: Converts requirements into formal feature specifications with testable Acceptance Criteria (`AC-1..n`).
   - **Software Architect**: Researches technical alternatives via web tools, selects stack with justification, defines ADRs, and writes compilable interface contract stubs.
   - **QA Test Engineer (Test-First)**: Writes exhaustive test suites *before* coding begins. Verified in sandbox with a deterministic RED-check (must compile and fail cleanly on `NotImplementedError`).
   - **Local Coders**: 100% local Ollama inference (`qwen3.5:9b` Metal GPU profile, \$0.00 model cost) working strictly against contracts and small task briefs (< 1500 tokens).
2. **Two-Pool Scheduling**:
   - **LocalPool**: Hard-locked to concurrency 1 to prevent Apple Silicon Metal GPU memory thrashing and thermal throttling.
   - **CloudPool**: Configurable concurrency $N$ for cloud specialists.
3. **Deterministic Self-Healing & Escalation**:
   - Quality gate failures automatically extract test names, line numbers, and truncated stack traces without LLM overhead.
   - Up to `MAX_FIX_ATTEMPTS` local repair turns.
   - Optional `--escalate-to-cloud` consults a cloud debugger for root-cause diagnosis if local attempts fail.
4. **Host-Side Web Research Tools**:
   - Safe `web_search` and `fetch_page` with strict SSRF defense (blocks loopback, RFC1918 private subnets, link-local, and cloud metadata `169.254.169.254`).
   - Built-in SearXNG (local default, zero keys) and Brave Search API support.
   - Results cached in SQLite FTS5 index (`.swarm/docs_index.db`) for offline search by local agents.
   - Citations verified and annotated with `[VERIFIED]` or `[UNVERIFIED]` status badges.
5. **Strict Security & Outbound Data Policy**:
   - First cloud call requires explicit operator confirmation (`.swarm/cloud_ack.json`).
   - Single outbound filter enforcing role-based allowed content classes, 64 KB per-call size limits, automatic API key/secret redaction, and a hard deny list (`.env*`, `*.pem`, `*.key`, `id_rsa`, `.git`, `.swarm`).
   - `--offline` instantly falls back all roles to local Ollama models.
   - All retrieved web content and gate feedback strictly delimited in `<untrusted_*>` tags.

---

## ⚡ Quick Start

### 1. Requirements & Platform Compatibility
- **Operating System**: macOS, Linux (Ubuntu/Debian/Arch/Fedora), or Windows (via WSL2).
- **Python**: Python 3.11+.
- **Git**: Installed and available in your `$PATH`.
- **Inference Options**:
  - **Cloud Inference (Recommended for best performance)**: SiliconFlow, DeepSeek, OpenAI, or Anthropic API key in `.env`.
  - **Local Inference (Optional)**: [Ollama](https://ollama.ai) installed with GPU acceleration (NVIDIA CUDA, AMD ROCm, or Apple Metal).
- **Sandboxing (Optional)**: Docker (Docker Desktop, Colima, or OrbStack) for sandboxed command execution, or `--unsafe-local-exec` for direct host development.

### 2. Setup
```bash
# Clone and enter repo
cd agent-swarm-se

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull recommended local model
ollama pull qwen3.5:9b

# Copy environment configuration
cp .env.example .env
```

### 3. Diagnostics & Health Probes
```bash
# Verify local Apple Silicon, Metal GPU offload, and Ollama capabilities
python main.py doctor

# Verify configured cloud LLM providers, API keys, JSON schema support, and pricing
python main.py providers doctor

# Estimate cloud financial costs for a goal before running
python main.py cost-estimate --goal "Add user authentication with JWT"
```

---

## 🛠️ Usage & Workflow

### 1. Run the Swarm
```bash
# Standard run on a repository
python main.py --repo /path/to/repo --goal "Add divide function with unit tests"

# Interactive requirements clarification with PM
python main.py --repo /path/to/repo --goal "Add payment checkout" --interactive

# Offline mode (forces all cloud roles to fallback to local Ollama)
python main.py --repo /path/to/repo --goal "Add math helper" --offline

# Safe mock dry-run (no Ollama, Docker, or real API calls)
python main.py --repo tests/fixtures/toy_repo --goal "Add modulo" --mock
```

### 2. Human Approval Gates
When running the full hybrid pipeline, execution pauses cleanly at deterministic human gates:

```bash
# Gate 1: Inspect and approve PM feature specification
cat .swarm/runs/<run_id>/artifacts/spec.md
python main.py approve-spec <run_id>

# Gate 2: Inspect and approve architecture design and interface contracts
cat .swarm/runs/<run_id>/artifacts/architecture.md
python main.py approve-design <run_id>
# (Or pass --auto-approve-design when starting the run for low-risk features)

# Resume execution after approval:
python main.py --resume <run_id>
```

### 3. PR Inspection & Final Merge
Upon task completion, a pull request artifact is generated at `.swarm/runs/<run_id>/PR.md`.

```bash
# Approve PR and merge integration branch into main:
python main.py approve --run-id <run_id>

# Reject PR with recorded audit reason:
python main.py reject --run-id <run_id> --reason "Needs more edge-case tests"
```

---

## ⚙️ CLI Flags & Options

| Flag | Default | Description |
|---|:---:|---|
| `--repo` | `.` | Target repository directory. |
| `--goal` | Positional | High-level software engineering goal. |
| `--interactive` | `False` | Clarify requirements interactively before planning. |
| `--offline` | `False` | Disable all cloud providers; fallback to local Ollama models. |
| `--cloud-sees-code` | `False` | Allow sending source code diffs to cloud roles (e.g. Reviewer). |
| `--log-payloads` | `False` | Save full outbound prompt text in audit logs (`outbound_audit.jsonl`). |
| `--auto-approve-design` | `False` | Automatically approve Gate 2 for low-risk runs. |
| `--waive-ac AC-ID` | `[]` | Waive specific acceptance criteria from blocking review (repeatable). |
| `--escalate-to-cloud` | `False` | Consult cloud debugger after local fix attempts fail. |
| `--resume RUN_ID` | `None` | Resume an interrupted or gate-paused run. |
| `--mock` | `False` | Run with MockProvider and MockSandbox (offline testing). |

---

## 🔒 Security, Privacy & Data Policy

All outbound calls to external cloud providers are governed by `config/policy.yaml`:

1. **API Keys**: Loaded exclusively from environment variables or `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `BRAVE_API_KEY`). Keys are never sent to prompts, sandboxes, logs, or git commits.
2. **First-Call Acknowledgment**: Before the first cloud API call per repository, the platform outputs an explicit disclosure listing providers, models, and content classes, recording operator approval in `.swarm/cloud_ack.json`.
3. **Hard Deny List**: Filenames matching `.env*`, `*.pem`, `*.key`, `*id_rsa*`, `.git/**`, `.swarm/**`, `config/providers.yaml`, and `config/policy.yaml` are strictly blocked from outbound cloud payloads.
4. **Outbound Content Classes**:
   - `pm` & `architect`: `goal`, `spec`, `repo_map`, `user_answers` only (no source code).
   - `qa`: `spec`, `architecture`, `interface_stubs`, `test_files`, `gate_logs`.
   - `reviewer`: `diffs` permitted only if `--cloud-sees-code` is passed.
   - `backend` & `tester`: Local Ollama only (100% offline).
5. **SSRF Guardrail**: `fetch_page` validates resolved IP addresses, blocking private RFC1918 subnets, loopbacks (`127.0.0.0/8`, `::1`), and cloud metadata (`169.254.169.254`).
6. **Financial Budgets**: Configurable cloud USD caps (`MAX_CLOUD_USD`). Costs are estimated and reserved before each call; runs pause cleanly if budgets are exceeded.

---

## 🧪 Testing

Run the full automated test suite (154 tests, zero network or Ollama calls required):

```bash
pytest
```

Tests verify:
- Multi-provider chat, tools, structured output, and streaming normalization.
- Outbound security filter, deny lists, secret redaction, and cloud acknowledgment.
- SSRF prevention, HTML sanitization, and SQLite FTS5 docs cache.
- PM, Architect, and QA schema validation and repair retries.
- Red-check test-first validation in sandbox.
- Two-pool scheduling and local concurrency 1 constraints.
- Deterministic failure diagnostics parsing and append-only attempt history.
- Pull request generation with full traceability matrix and cost accounting.
