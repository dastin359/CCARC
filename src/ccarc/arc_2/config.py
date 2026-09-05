"""Run configuration for the Claude Code harness variant."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

#: Built-in tools the solver may use. Empty means **do not pass ``--tools``** —
#: the solver gets Claude Code's full default surface, minus the deny list.
#:
#: This is the point of the harness: "Claude Code as harness" is only an honest
#: test if the agent gets Claude Code, not a hand-picked six-tool subset that
#: withholds sub-agents, workflows, skills and background tasks — capability
#: the product actually ships and that a real user would have.
DEFAULT_TOOLS: tuple[str, ...] = ()

#: Tools **pre-approved** so a headless run never stalls on a permission prompt.
#:
#: ``--allowedTools`` does two jobs at once: it restricts the surface *and* it
#: grants permission. Dropping it to open the surface also removes the grant —
#: `acceptEdits` auto-approves file writes but not arbitrary Bash, so without a
#: grant every `python explore/foo.py` is denied and the run produces nothing.
#: The surface is opened by omitting ``--tools``; permission is granted
#: here, explicitly.
DEFAULT_ALLOWED_TOOLS: tuple[str, ...] = (
    "Bash", "Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit",
    "Task", "Workflow", "Skill", "ToolSearch", "Monitor", "SendMessage",
    "TodoWrite", "TaskCreate", "TaskGet", "TaskList", "TaskOutput",
    "TaskStop", "TaskUpdate", "EnterWorktree", "ExitWorktree",
    "ListSkills", "SearchSkills", "SuggestSkills", "ReportFindings",
)

#: Denied outright, in three groups. Everything else the product offers is
#: available to the solver, including ``Task`` (sub-agents) and ``Workflow``.
#:
#: 1. **Research.** ARC requires no external knowledge and a network answer
#:    would contaminate the benchmark outright.
#: 2. **Reaching the network by another door.** Registry, plugin and connector
#:    lookups are not "research", but they are outbound calls, and a benchmark
#:    that claims no network access has to mean it.
#: 3. **Escaping the run.** Publishing, notifying the user, or scheduling work
#:    that outlives the run all break the containment the scoring depends on —
#:    a solver must not be able to emit anything except into its workspace.
DEFAULT_DISALLOWED_TOOLS: tuple[str, ...] = (
    # 1 — research
    "WebSearch", "WebFetch",
    # 2 — network by another door
    "SearchMcpRegistry", "SearchPlugins", "SuggestPluginInstall",
    "SuggestConnectors", "ListConnectors", "ListPlugins",
    "ListMcpResourcesTool", "ReadMcpResourceTool", "ReadMcpResourceDirTool",
    # 3 — escaping the run
    "Artifact", "SendUserFile", "PushNotification",
    "CronCreate", "CronDelete", "CronList", "ScheduleWakeup",
    "ShowOnboardingRolePicker",
)


@dataclass
class CCRunConfig:
    """Everything that shapes one CC-harness solve attempt.

    Persisted verbatim into each run's ``result.json`` so a run can be
    reproduced from its own record.
    """

    # ── model / agent loop ───────────────────────────────────────────────
    model: str = "opus"
    effort: str = "high"
    """Claude Code reasoning effort: low | medium | high | xhigh | max."""

    # ── budgets ──────────────────────────────────────────────────────────
    max_iterations: int = 12
    """Formal `gate.py submit` calls allowed. Exploration is unbudgeted."""
    best_effort_iterations: int = 2
    """Trailing iterations during which the train-100% requirement is lifted."""
    max_budget_usd: float | None = None
    """Passed to `claude --max-budget-usd`; None leaves spend uncapped."""
    wall_clock_timeout_s: float = 5400.0
    solve_timeout_s: float = 60.0
    """Per-submission ceiling on running solve() across all inputs."""

    # ── task presentation ────────────────────────────────────────────────
    max_test_predictions: int = 2
    visual: bool = True
    """Render PNGs of every grid so the agent can Read them as images."""
    inline_grids: bool = True
    """Inline the text grids in the opening prompt (vs. file reference only)."""
    min_hypothesis_chars: int = 300
    """Floor enforced by the gate, standing in for 'be exhaustive'."""

    # ── harness isolation ────────────────────────────────────────────────
    permission_mode: str = "acceptEdits"
    """Claude Code permission mode.

    `acceptEdits` auto-approves file writes and common filesystem commands;
    everything else the solver needs is covered by `tools` via `--allowedTools`.

    Not `bypassPermissions`: Claude Code maps it to
    `--dangerously-skip-permissions`, which the CLI refuses outright when the
    process is running as root — the normal case for a containerised research
    harness. `dontAsk` is the stricter alternative for locked-down runs: it
    denies anything not explicitly allowed rather than prompting.
    """
    tools: tuple[str, ...] = DEFAULT_TOOLS
    allowed_tools: tuple[str, ...] = DEFAULT_ALLOWED_TOOLS
    disallowed_tools: tuple[str, ...] = DEFAULT_DISALLOWED_TOOLS
    stable_system_prompt: bool = True
    """Pass `--exclude-dynamic-system-prompt-sections`.

    Every task runs in its own workspace, so cwd differs per task — and cwd sits
    in Claude Code's default system prompt. That changes the cached prefix on
    every task and defeats cross-task prompt-cache reuse entirely: a second
    workspace writes byte-identical cache to the first rather than reading it.

    What this recovers is the system-prompt prefix across tasks, a few thousand
    tokens. The bulk of a run's cache traffic is within-task, as each turn
    writes the growing conversation, so the flag restores a defeated mechanism
    rather than being a material cost lever.

    The flag moves cwd, env info, memory paths and git status into the first
    user message instead. It applies only alongside the default system prompt,
    which is what this harness uses (`--append-system-prompt`, not
    `--system-prompt`).
    """

    setting_sources: str = "project"
    """Which Claude Code settings sources to load. 'project' keeps the
    workspace's own .claude/settings.json (the compaction hook) while ignoring
    whatever the host user has configured."""
    bare: bool = False
    """Run `claude --bare` for maximum reproducibility. Requires
    ANTHROPIC_API_KEY — bare mode skips OAuth and keychain reads."""
    extra_cli_args: tuple[str, ...] = ()

    # ── bookkeeping ──────────────────────────────────────────────────────
    #: Named prompt components to remove, e.g. ``("doctrine",)`` or
    #: ``("workspace:Rival readings",)``. Recorded in ``to_dict`` so a run
    #: record states its own manipulation; an unknown name raises.
    ablate: tuple[str, ...] = ()

    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.max_iterations = max(1, int(self.max_iterations))
        self.best_effort_iterations = max(0, min(int(self.best_effort_iterations), self.max_iterations))
        self.max_test_predictions = max(1, min(2, int(self.max_test_predictions)))
        self.tools = tuple(self.tools)
        self.allowed_tools = tuple(self.allowed_tools)
        self.disallowed_tools = tuple(self.disallowed_tools)
        self.extra_cli_args = tuple(self.extra_cli_args)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CCRunConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})
