"""``ccarc arc-2`` — command line for the Claude Code harness variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import list_tasks

from .config import CCRunConfig
from .runner import (
    default_event_printer,
    discover_runs,
    rescore_run,
    resume_task,
    run_batch,
    run_task,
)
from .scoring import aggregate, format_aggregate
from .trace import collect_trace, format_trace

DEFAULT_OUT_DIR = "cc_runs"


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Where run directories are written.")
    parser.add_argument("--dataset-root", default=None, help="ARC dataset root (defaults to $ARC_DATA_ROOT).")
    parser.add_argument("--split", default="public_eval", help="Dataset split (default: public_eval).")
    parser.add_argument("--model", default="opus", help="Model alias or id passed to `claude --model`.")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-iterations", type=int, default=12, help="Budgeted `gate.py submit` calls.")
    parser.add_argument(
        "--best-effort-iterations",
        type=int,
        default=2,
        help="Trailing iterations over which the train-100%% requirement is lifted.",
    )
    parser.add_argument("--max-test-predictions", type=int, default=2, choices=[1, 2])
    parser.add_argument("--max-budget-usd", type=float, default=None, help="Per-task spend cap.")
    parser.add_argument(
        "--timeout", type=float, default=5400.0,
        help="Per-task wall-clock limit in seconds. On expiry a train-perfect "
             "submission that was never accepted is salvaged and scored.",
    )
    parser.add_argument("--solve-timeout", type=float, default=60.0, help="Per-submission solve() limit.")
    parser.add_argument("--no-visual", action="store_true", help="Skip PNG rendering of the grids.")
    parser.add_argument("--no-inline-grids", action="store_true", help="Keep the grids out of the opening prompt.")
    parser.add_argument(
        "--permission-mode",
        default="acceptEdits",
        choices=["acceptEdits", "dontAsk", "bypassPermissions", "auto", "manual", "plan"],
        help="Claude Code permission mode. bypassPermissions is refused when running as root.",
    )
    parser.add_argument(
        "--bare",
        action="store_true",
        help="Run `claude --bare` for maximum reproducibility (requires ANTHROPIC_API_KEY).",
    )
    parser.add_argument("--setting-sources", default="project")
    parser.add_argument(
        "--no-stable-system-prompt",
        action="store_true",
        help="Keep cwd and env info in the system prompt (breaks cross-task prompt-cache reuse).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing workspace.")
    parser.add_argument(
        "--ablate",
        action="append",
        default=[],
        metavar="TARGET",
        help=(
            "Remove a named prompt component, repeatable. 'doctrine' drops the doctrine "
            "from the appended system prompt; 'workspace:<Section>' drops a section from "
            "the workspace CLAUDE.md. An unknown name is an error, and the choice is "
            "recorded in the run's config so the manipulation is reproducible from it."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Do not trace agent events to stdout.")


def _config_from_args(args: argparse.Namespace) -> CCRunConfig:
    return CCRunConfig(
        model=args.model,
        effort=args.effort,
        max_iterations=args.max_iterations,
        best_effort_iterations=args.best_effort_iterations,
        max_test_predictions=args.max_test_predictions,
        max_budget_usd=args.max_budget_usd,
        wall_clock_timeout_s=args.timeout,
        solve_timeout_s=args.solve_timeout,
        visual=not args.no_visual,
        inline_grids=not args.no_inline_grids,
        permission_mode=args.permission_mode,
        bare=args.bare,
        setting_sources=args.setting_sources,
        stable_system_prompt=not args.no_stable_system_prompt,
        ablate=tuple(getattr(args, "ablate", ()) or ()),
    )


def _print_task_result(record: dict) -> None:
    score = record.get("score") or {}
    status = "SOLVED" if score.get("solved") else ("partial" if score.get("num_solved") else "unsolved")
    print(
        f"  {record.get('task_id')}: {status}  "
        f"{score.get('num_solved', 0)}/{score.get('num_test', 0)} test examples  "
        f"{record.get('iterations_used', 0)}/{record.get('max_iterations', '?')} iterations"
        + (f"  ${float(record['cost_usd']):.3f}" if record.get("cost_usd") is not None else "")
    )
    if record.get("error"):
        print(f"    error: {record['error']}")
    if (record.get("integrity") or {}).get("suspected"):
        for item in record["integrity"]["evidence"]:
            print(f"    INTEGRITY: {item}")
    for finding in record.get("hardcoding_findings") or []:
        print(f"    overfit signal: {finding}")


def cmd_run(args: argparse.Namespace) -> int:
    record = run_task(
        args.task,
        out_dir=args.out_dir,
        config=_config_from_args(args),
        dataset_root=args.dataset_root,
        dataset_split=args.split,
        event_callback=None if args.quiet else default_event_printer,
        overwrite=args.overwrite,
    )
    _print_task_result(record)
    return 0 if record.get("accepted") and not record.get("error") else 1


def cmd_workspace(args: argparse.Namespace) -> int:
    record = run_task(
        args.task,
        out_dir=args.out_dir,
        config=_config_from_args(args),
        dataset_root=args.dataset_root,
        dataset_split=args.split,
        overwrite=args.overwrite,
        launch=False,
    )
    workspace = Path(record["workspace"])
    run_dir = workspace.parent
    print(f"workspace     : {workspace}")
    print(f"system prompt : {run_dir / 'system_prompt.md'}")
    print(f"opening prompt: {run_dir / 'initial_prompt.md'}")
    print()
    print("Point any Claude Code agent at that directory. From inside it:")
    print("  python gate.py status | submit | accept")
    print()
    print("Then inspect and score it with:")
    print(f"  ccarc arc-2 trace {run_dir}")
    print(f"  ccarc arc-2 score {args.out_dir}")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    tasks = list(args.tasks or [])
    if args.all:
        tasks = list_tasks(split=args.split, dataset_root=args.dataset_root)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("No tasks selected. Pass --tasks, or --all.")
        return 2

    results = run_batch(
        tasks,
        out_dir=args.out_dir,
        config=_config_from_args(args),
        dataset_root=args.dataset_root,
        dataset_split=args.split,
        event_callback=None if args.quiet else default_event_printer,
        overwrite=args.overwrite,
        resume_incomplete=args.resume_incomplete,
        on_task_done=_print_task_result,
    )

    summary = aggregate(results)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8"
    )
    print()
    print(format_aggregate(summary, results))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    if not out_dir.is_dir():
        print(f"No such run directory: {out_dir}")
        return 2

    results = []
    for run_dir in discover_runs(out_dir):
        # Recompute from the workspace by default: the workspace is the record,
        # and result.json is a cache that goes stale the moment an agent keeps
        # working — including a workspace prepared by `cc workspace` and then
        # driven by a sub-agent, whose cached record predates all of the work.
        try:
            if args.cached and (run_dir / "result.json").is_file():
                results.append(json.loads((run_dir / "result.json").read_text(encoding="utf-8")))
            else:
                results.append(rescore_run(run_dir, dataset_root=args.dataset_root, dataset_split=args.split))
        except Exception as exc:  # noqa: BLE001 - one unreadable run must not sink the sweep
            print(f"  {run_dir.name}: could not score ({type(exc).__name__}: {exc})")
            results.append({"task_id": run_dir.name, "error": f"{type(exc).__name__}: {exc}"})

    if not results:
        print(f"No runs found under {out_dir}")
        return 2

    summary = aggregate(results)
    (out_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8"
    )
    print(format_aggregate(summary, results))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    failures = 0
    for run_dir in args.run_dirs:
        record = resume_task(
            run_dir,
            dataset_root=args.dataset_root,
            dataset_split=args.split,
            event_callback=None if args.quiet else default_event_printer,
        )
        if record.get("resume_skipped"):
            print(f"  {record.get('task_id')}: skipped ({record['resume_skipped']})")
        _print_task_result(record)
        failures += 0 if record.get("accepted") else 1
    return 0 if failures == 0 else 1


def cmd_trace(args: argparse.Namespace) -> int:
    targets = [Path(p) for p in args.paths]
    if not targets:
        out_dir = Path(DEFAULT_OUT_DIR).resolve()
        targets = sorted(p for p in out_dir.iterdir() if (p / "workspace").is_dir()) if out_dir.is_dir() else []
    if not targets:
        print("No workspaces found. Pass one or more run directories.")
        return 2

    for index, target in enumerate(targets):
        if index:
            print("\n" + "=" * 72 + "\n")
        try:
            print(format_trace(collect_trace(target), verbose=args.verbose))
        except Exception as exc:  # noqa: BLE001 - a bad directory should not stop the sweep
            print(f"{target}: could not read a trace ({type(exc).__name__}: {exc})")
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the `cc` sub-commands to an existing parser."""
    sub = parser.add_subparsers(dest="cc_command", required=True)

    run = sub.add_parser("run", help="Solve one task with Claude Code as the agent loop.")
    run.add_argument("task", help="Task id or path to a task JSON.")
    _add_common_arguments(run)
    run.set_defaults(func=cmd_run)

    workspace = sub.add_parser(
        "workspace",
        help="Prepare a workspace and its prompts without launching a solver.",
    )
    workspace.add_argument("task", help="Task id or path to a task JSON.")
    _add_common_arguments(workspace)
    workspace.set_defaults(func=cmd_workspace)

    batch = sub.add_parser("batch", help="Solve many tasks sequentially.")
    batch.add_argument("--tasks", nargs="*", default=[], help="Task ids.")
    batch.add_argument("--all", action="store_true", help="Every task in the split.")
    batch.add_argument("--limit", type=int, default=None, help="Cap the number of tasks.")
    batch.add_argument(
        "--resume-incomplete",
        action="store_true",
        help=(
            "Make the batch restartable: skip tasks that already have a result, "
            "resume tasks whose solver was killed mid-run, and only start the rest. "
            "Cannot be combined with --overwrite."
        ),
    )
    _add_common_arguments(batch)
    batch.set_defaults(func=cmd_batch)

    resume = sub.add_parser(
        "resume",
        help="Relaunch a solver into an existing workspace that was never accepted.",
    )
    resume.add_argument("run_dirs", nargs="+", help="Run directories to resume.")
    resume.add_argument("--dataset-root", default=None)
    resume.add_argument("--split", default="public_eval")
    resume.add_argument("--quiet", action="store_true")
    resume.set_defaults(func=cmd_resume)

    trace = sub.add_parser(
        "trace",
        help="Reconstruct what a solver agent did: verification density, submissions, refusals.",
    )
    trace.add_argument("paths", nargs="*", help="Run or workspace directories (default: every run in cc_runs).")
    trace.add_argument("-v", "--verbose", action="store_true", help="Include hypothesis evolution and the audit.")
    trace.set_defaults(func=cmd_trace)

    score = sub.add_parser("score", help="Aggregate finished runs in a directory.")
    score.add_argument("out_dir", nargs="?", default=DEFAULT_OUT_DIR)
    score.add_argument(
        "--cached",
        action="store_true",
        help="Trust each run's stored result.json instead of recomputing from its workspace.",
    )
    score.add_argument("--dataset-root", default=None)
    score.add_argument("--split", default="public_eval")
    score.set_defaults(func=cmd_score)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ccarc arc-2",
        description="Drive a coding agent at ARC-AGI-2, inside a guardrail that "
                    "makes the result checkable.",
    )
    add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
