# DevSwarm

DevSwarm is an autonomous multi-agent development tool. Give it a task, and instead of having a single AI try to do everything at once, DevSwarm coordinates a pipeline of specialized agents:

1. **Product Manager**: Clarifies the goal and writes a spec with acceptance criteria.
2. **Architect**: Designs the approach and defines interface contracts.
3. **QA Engineer**: Writes failing tests *first* (TDD) before implementation starts.
4. **Backend / Frontend**: Writes the actual code in isolated Git worktrees.
5. **Quality Gates**: Runs real compiler, linter (`ruff`), test suite (`pytest`), and AST security checks.
6. **Senior Reviewer**: Audits the diffs, checks test coverage against the spec, and generates a review (`PR.md`).

The project works with both **Cloud LLMs** (SiliconFlow, DeepSeek, OpenAI, Anthropic) and **Local LLMs** (Ollama on macOS, Linux, or Windows WSL2).

---

## How It Works

```
                        You (Goal Prompt)
                               │
                               ▼
                        [Product Manager]
                               │  spec.md with Acceptance Criteria
                               ▼
                       [Software Architect]
                               │  architecture.md & contract stubs
                               ▼
                          [QA Engineer]
                               │  Failing test suite (RED-check verified)
                               ▼
                        [Specialist Coders]
                               │  Isolated Git worktrees per task
                               ▼
                   [Deterministic Quality Gates]
                   compileall ➔ ruff ➔ pytest ➔ AST security
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
          Passed Gates                   Gates Failed
                │                             │
                ▼                             ▼
        Git Merge Commit               Self-Healing Loop
                │                   (Fixes errors with diagnostics)
                ▼
         [Senior Reviewer]
                │
                ▼
         Final Pull Request (PR.md)
                │
                ▼
          Approved & Merged
```

---

## Why DevSwarm?

- **Real Quality Gates, Not Hallucinations**: An agent isn't done just because it outputs text. Tasks must compile, pass formatting/lint rules, and pass actual unit test runs in the terminal.
- **Isolated Git Worktrees**: Agents don't touch your main working tree directly. Each task runs in its own checkout (`.swarm/worktrees/<task>`), so broken code never pollutes your repository.
- **Self-Healing Diagnostics**: When a test or lint fails, the runner parses the exact failure lines and traceback, passing them back to the agent to fix the issue automatically.
- **Zero-Trust Security**: No `eval()` or dangerous primitives. File paths, secrets (`.env`, SSH keys), and internal IPs are guarded by strict outbound filters.
- **Cross-Platform**: Runs on macOS, Linux (Ubuntu/Debian), and Windows (WSL2).

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Git
- An API key for your chosen provider (e.g. SiliconFlow, DeepSeek, OpenAI, Anthropic) or local Ollama.

### 2. Installation

```bash
git clone https://github.com/<your-username>/devswarm_se.git
cd devswarm_se

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure your environment
cp .env.example .env
```

Open `.env` and configure your API key (for example `SILICONFLOW_API_KEY=...` or `OPENAI_API_KEY=...`).

### 3. Run on a Project

Point DevSwarm at any target repository and tell it what you want built:

```bash
python main.py --repo /path/to/your/project --goal "Implement a rate limiter middleware with tests"
```

Useful flags:
- `--interactive`: Lets the PM ask you clarifying questions before generating the spec.
- `--roles pm,architect,qa,reviewer,backend,tester`: Pick which roles to run.
- `--cloud-sees-code`: Allows the reviewer agent to inspect diffs when using cloud models.
- `--unsafe-local-exec`: Runs quality checks directly on your host shell instead of Docker.

---

## Human Approval & Control

DevSwarm gives you full control at every stage:

```bash
# Check the status of current runs
python main.py status

# Inspect the generated PR
cat .swarm/runs/<run_id>/PR.md

# Approve and merge the changes into main
python main.py approve --run-id <run_id>

# Reject with feedback
python main.py reject --run-id <run_id> --reason "Add more edge case tests"

# Emergency kill switch (halts all agent loops immediately)
python main.py stop
```

---

## Project Structure

```text
├── config/
│   ├── gates.yaml          # Quality gate commands (compileall, ruff, pytest)
│   ├── models.yaml         # Model capabilities, context limits, and token pricing
│   ├── policy.yaml         # Security rules and redaction filters
│   ├── providers.yaml      # Cloud & local LLM endpoint configurations
│   └── roles.yaml          # Agent role definitions, budgets, and tool access
├── devswarm/
│   ├── agents/             # Specialist prompts, context compaction, and loop logic
│   ├── dev_orchestrator.py # Swarm execution pipeline and DAG scheduler
│   ├── gates.py            # Quality gate runner (compile, lint, test, security)
│   ├── platform/           # Worktree isolation, token budgets, guards, and sandboxing
│   ├── providers/          # OpenAI, Anthropic, SiliconFlow, and Ollama adapters
│   └── reviewer.py         # PR review generator and diff evaluation
├── main.py                 # CLI entry point
└── tests/                  # Automated test suite (162 tests)
```

---

## Running Tests

DevSwarm comes with a comprehensive test suite (162 unit & integration tests) covering adapters, prompt parsing, budget reservation, worktrees, and quality gates:

```bash
pytest
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
