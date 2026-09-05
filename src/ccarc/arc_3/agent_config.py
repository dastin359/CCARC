"""How the coding agent the harness launches is configured.

The agent runs as a subprocess of the driver, so its tool surface and its
permission mode are set on the command line rather than in this package. The
values live here because they describe the agent, not the game.
"""
from __future__ import annotations

import os

#: Offered to the solver. Everything the product exposes that is useful for
#: playing, including sub-agents and workflows.
DEFAULT_ALLOWED_TOOLS: tuple[str, ...] = (
    "Bash", "Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit",
    "Task", "Workflow", "Skill", "ToolSearch", "Monitor", "SendMessage",
    "TodoWrite", "TaskCreate", "TaskGet", "TaskList", "TaskOutput",
    "TaskStop", "TaskUpdate", "EnterWorktree", "ExitWorktree",
    "ListSkills", "SearchSkills", "SuggestSkills", "ReportFindings",
)

#: Denied outright, in three groups.
#:
#: 1. **Research.** ARC requires no external knowledge and a network answer
#:    would contaminate the benchmark.
#: 2. **Network by another door.** Registry and connector tools reach outside
#:    the run without naming themselves as network access.
#: 3. **Escaping the run.** Anything that publishes, notifies, or schedules
#:    work beyond the process the harness is measuring.
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


def running_as_root() -> bool:
    getuid = getattr(os, "geteuid", None)
    return bool(getuid and getuid() == 0)


def resolve_permission_mode(mode: str) -> str:
    """Substitute a permission mode the CLI will accept.

    ``bypassPermissions`` maps to a flag Claude Code refuses when running as
    root, and the refusal arrives as an empty stream with a one-line stderr. A
    containerised harness is usually root, so trade it for ``acceptEdits``,
    which does the same job here because every tool the solver needs is
    already allow-listed.
    """
    if mode == "bypassPermissions" and running_as_root():
        return "acceptEdits"
    return mode
