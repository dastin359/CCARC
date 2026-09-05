"""Mount-namespace confinement for a solver: outside its workspace, `ENOENT`.

**Mechanism: nested user namespaces**, the same dance `bwrap` does:

1. An outer namespace maps root — required for `mount` — and lays tmpfs over
   every hidden tree, with the solver's own directories *held* first via
   ``--rbind`` to a staging tmpfs and re-bound on top afterwards.
2. An inner namespace maps back to the real uid before ``exec``, because the
   CLI refuses ``--dangerously-skip-permissions`` under root; the nesting
   resolves the conflict.

**What survives, and why exactly these:**

- the ATTEMPT directory, read-write — the workspace and its sidecars,
  including the session jar the client keeps one level above the workspace;
- ``src/ccarc``, read-only in effect — the solver imports the package
  (strict mode keeps the staged copy in the attempt directory instead);
- the repo's ``.venv`` — the interpreter with numpy that PATH points at;
- the solver's own ``~/.claude/projects/<key>`` — its transcripts, which
  ``--resume`` (every nudge arm) replays. Held and re-bound rather than
  relocated, so the CLI's own auth files stay where the CLI expects them.

The network is handled by the companion module: :mod:`ccarc.guardrail.network`
gives the solver a network namespace whose only route is the loopback socket
the shim answers on.

**What disappears:** the scratch tree (the driver log, the ARC key file, the
shared-card cookies, every other game's workspace and every previous
sweep's), the repo's non-package directories and ``.git``, session
scratchpads under ``/tmp/claude-*``, and everyone else's
``~/.claude/projects`` entries.

**Fail loud, never open.** A hidden path that does not exist is skipped by
the *builder* (and stated), never by the script — a typo'd path silently
"hidden" by `|| true` at mount time is a confinement with a gap in it, reported
as complete. When namespaces are unavailable the launch refuses rather than
running unconfined.

On by default; ``CCARC3_CONFINE=0`` stands it down (read by
:func:`ccarc.arc_3.session._launch`).
"""

# **Why this paragraph is a comment.** This module is staged onto the solver's
# own PYTHONPATH, and a module docstring is one attribute access from the
# namespace the workspace hands it. `staging` strips comments and keeps
# docstrings, so the argument for the guardrail is written on the side of that line
# the solver cannot read.
#
# Prevention, not a rule: under this module a read outside the workspace is a
# missing file rather than a forbidden one. The mount table is the boundary, so
# confinement is a property of the launch command rather than a claim about the
# run.


from __future__ import annotations

import json
import os
import time
import re
import shlex
import shutil
import tempfile
import subprocess
from collections.abc import Mapping  # noqa: F401
from pathlib import Path

__all__ = ["available", "hidden_paths", "build_command", "CONFINE_ENV"]

CONFINE_ENV = "CCARC3_CONFINE"

_REPO = Path(__file__).resolve().parents[2].parent


def available() -> str:
    """"" if confinement cannot work here, else a short how.

    Probes the real mechanism — nested unshare with a mount — because the
    binary existing proves nothing about the kernel's userns policy, and a
    launch-time surprise costs a game.
    """
    if not shutil.which("unshare"):
        return ""
    try:
        probe = subprocess.run(
            ["unshare", "--user", "--mount", "--map-root-user", "/bin/sh", "-c",
             "mount -t tmpfs none /mnt && exec unshare --user --map-user=1 true"],
            capture_output=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return "unshare" if probe.returncode == 0 else ""


def _project_key(workspace: Path) -> str:
    """The CLI's per-cwd transcript directory name for this workspace.

    **Every non-alphanumeric byte becomes a dash, underscores included**, and
    the derivation must match the CLI's exactly.

    Getting this wrong is silent data loss, not an error: `hide_after` lays a
    tmpfs over `projects` and `keep_last` re-binds this one key, so a
    mis-derived key means the bind lands on a directory the CLI never writes
    to, the CLI writes to a neighbour on the tmpfs, and the transcript dies
    with the mount namespace -- every later `--resume` then reports no such
    conversation.
    """
    return re.sub(r"[^A-Za-z0-9]", "-", str(workspace.resolve()))



def config_dir(env: "Mapping[str, str] | None" = None) -> Path:
    """The CLI's config directory for the run being launched.

    `CLAUDE_CONFIG_DIR` relocates BOTH the config directory and the account file
    -- the CLI resolves the latter as
    ``join(process.env.CLAUDE_CONFIG_DIR || homedir(), ".claude<suffix>.json")``.
    A launch with the variable set draws on that directory's account and no
    other.

    This must be derived, never hardcoded to ``~/.claude``. A config directory
    that is not in the keep list is hidden by the strict-mode tmpfs and the
    launch **fails**, which is the correct failure; keeping ``~/.claude`` as
    well would hand the run credentials it was not launched with.
    """
    # **Read the LAUNCH's env, not the process's.** The driver runs games on
    # threads and each one gets its own config directory, so a module-level
    # `os.environ` lookup would hand every thread the same answer. Falls back
    # to the process env for callers with no launch env.
    raw = (env or {}).get("CLAUDE_CONFIG_DIR") or os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(raw).expanduser().resolve() if raw else Path.home() / ".claude"


def per_run_config(into: Path, *, base: Path | None = None) -> Path:
    """Give one run its own CLI config directory, and return it.

    **A shared config directory does not survive concurrent solvers.** The CLI
    rewrites `.claude.json` as it starts, so every launch is a writer; two
    concurrent launches can leave the file truncated to zero bytes, and every
    launch after that exits before its first action. The only reliable fix is
    that no two launches share one.

    Copies the credential and account files into ``into`` -- which lives inside
    the run's own attempt tree, so the guardrail keeps it with no extra keep-list
    entry and it dies with the run. `settings.json` is not copied: a status
    line configured there is executed inside the guardrail on every render, and it
    is one more writer in the directory.

    `hasCompletedOnboarding` is forced, because a config directory without it
    sends an interactive CLI into the onboarding flow instead of the task.
    """
    base = Path(base) if base else config_dir()
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    creds = base / ".credentials.json"
    if creds.exists():
        shutil.copy2(creds, into / ".credentials.json")
        os.chmod(into / ".credentials.json", 0o600)
    acct_src = base / ".claude.json"
    if not acct_src.exists() or acct_src.stat().st_size < 1024:
        # A base account file under 1 KB is a stub, not a usable account
        # file; prefer the known-good copy beside it.
        kg = base / ".claude.json.known-good"
        if kg.exists():
            acct_src = kg
    try:
        acct = json.loads(acct_src.read_text()) if acct_src.exists() else {}
    except Exception:                                   # noqa: BLE001
        acct = {}
    acct["hasCompletedOnboarding"] = True
    acct.setdefault("lastOnboardingVersion", "2.1.233")
    acct.pop("statusLine", None)
    dst = into / ".claude.json"
    fd, tmp = tempfile.mkstemp(dir=str(into))
    with os.fdopen(fd, "w") as fh:
        json.dump(acct, fh)
    os.replace(tmp, dst)
    os.chmod(dst, 0o600)
    shutil.copy2(dst, into / ".claude.json.known-good")
    os.chmod(into / ".claude.json.known-good", 0o600)
    return into


def harvest_credentials(from_dir: Path, into: Path) -> str:
    """Copy a refreshed credential back to the directory the run was seeded from.

    The failure mode this guards: a long run refreshes and receives a NEW
    rotating refresh token, invalidating the copy still in the source
    directory; the next attempt then copies a dead credential, the refused
    refresh blanks the file, and the run dies with the same signature a hard
    game produces.

    Harvest only. It never writes a blank over a good credential, and never an
    older one over a newer, because both are how the failure propagates rather
    than stops.
    """
    src = Path(from_dir) / ".credentials.json"
    dst = Path(into) / ".credentials.json"
    if not src.exists():
        return "no credential in the run directory"
    try:
        got = (json.loads(src.read_text()).get("claudeAiOauth") or {})
    except Exception:                                   # noqa: BLE001
        return "the run's credential does not parse; leaving the source alone"
    if not (got.get("accessToken") or "").strip():
        # The CLI blanks its own copy when a refresh fails. Copying that back
        # would spread the failure to every future run from this directory.
        return "the run's credential is blank (a failed refresh); not harvesting"
    have = {}
    if dst.exists():
        try:
            have = json.loads(dst.read_text()).get("claudeAiOauth") or {}
        except Exception:                               # noqa: BLE001
            have = {}
    if (got.get("expiresAt") or 0) <= (have.get("expiresAt") or 0):
        return "the source credential is at least as fresh; nothing to harvest"
    Path(into).mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(into))
    with os.fdopen(fd, "w") as fh:
        fh.write(src.read_text())
    os.replace(tmp, dst)
    os.chmod(dst, 0o600)
    return (f"harvested a refreshed credential; expiry moves to "
            f"{got.get('expiresAt')}")


def credential_is_usable(base: Path | None = None) -> str:
    """"" if the credential can plausibly start a run, else why it cannot.

    Checked BEFORE a launch, because the alternative is discovering it after:
    a dead credential produces "solver exited 1 after 0 actions", which is
    indistinguishable from a hard game and is counted as an attempt.
    """
    p = (Path(base) if base else config_dir()) / ".credentials.json"
    if not p.exists():
        return f"no credential file at {p}"
    try:
        t = json.loads(p.read_text()).get("claudeAiOauth") or {}
    except Exception as exc:                            # noqa: BLE001
        return f"{p} does not parse: {exc}"
    if not (t.get("accessToken") or "").strip():
        return f"{p} holds a blank access token (a failed refresh wrote over it)"
    if not (t.get("refreshToken") or "").strip():
        return f"{p} holds no refresh token, so a long run cannot survive expiry"
    return ""


# The CLI is the only thing that knows how to refresh, so the only honest test
# of a credential is to make it try: drive `claude -p` on the cheapest model.
# Two properties matter. It runs against a THROWAWAY COPY, because a refused refresh
# blanks the credential file and the base must not be what gets blanked; and it
# harvests afterwards, because the probe rotates the token and the base would
# otherwise keep the one the rotation just killed.
_PROBE_MODEL = "claude-haiku-4-5-20251001"
_PROBE_REFUSED = re.compile(
    r"authenticat|oauth|401|invalid[_ ]token|token (has )?expired|session expired|"
    r"please run .{0,20}login",
    re.IGNORECASE)


def _cli_binary() -> str:
    """The `claude` executable.

    Resolved here rather than imported from `session`, which imports this
    module. The two resolutions must agree.
    """
    return os.environ.get("CLAUDE_BINARY") or shutil.which("claude") or "claude"


def credential_probe(base: Path | None = None, *, lead_s: float = 1800,
                     timeout: float = 120, model: str = _PROBE_MODEL) -> str:
    """"" if the credential really works, else why it does not. Spends no action.

    :func:`credential_is_usable` reads the file, and a file can be perfectly
    well-formed and refused by the server -- an expired access token whose
    refresh token has been superseded passes every static check and is then
    refused at launch. That is the failure the static check cannot see.

    **Only when the token cannot be trusted to be live.** An access token with
    time left on it needs no refresh and will work, so probing it would spend a
    model call and a rotation to learn nothing. The probe fires when the token
    has expired or is within ``lead_s`` of expiring -- which is exactly when the
    refresh path is about to be exercised for real.

    A long run that crosses its expiry mid-flight is not this function's
    problem: `_run_live` repairs a refused credential and carries on.
    """
    base = Path(base) if base else config_dir()
    why = credential_is_usable(base)
    if why:
        return why
    try:
        token = json.loads((base / ".credentials.json").read_text())["claudeAiOauth"]
        expires = float(token.get("expiresAt") or 0) / 1000
    except Exception:                                   # noqa: BLE001
        expires = 0.0
    if expires and expires - time.time() > lead_s:
        return ""                       # live, with room to spare

    scratch = Path(tempfile.mkdtemp(prefix="ccarc3-credprobe-"))
    try:
        cfg = per_run_config(scratch / "cfg", base=base)
        env = dict(os.environ)
        env["CLAUDE_CONFIG_DIR"] = str(cfg)
        # Inherited, it would make the probe write its transcript under the
        # caller's session id and file its events as the caller's.
        env.pop("CLAUDE_CODE_SESSION_ID", None)
        try:
            done = subprocess.run(
                [_cli_binary(), "-p", "Reply with the single word: ok",
                 "--model", model],
                capture_output=True, text=True, timeout=timeout,
                env=env, cwd=str(scratch))
        except subprocess.TimeoutExpired:
            return f"the CLI did not answer within {timeout:.0f}s using {base}"
        except OSError as exc:
            return f"could not run the CLI to test {base}: {exc}"
        said = f"{done.stdout}\n{done.stderr}".strip()
        if done.returncode != 0 or _PROBE_REFUSED.search(said):
            first = next((l for l in said.splitlines() if l.strip()), "no output")
            return f"the server refused the credential in {base}: {first[:160]}"
        # The probe rotated the token; the base must end up holding the new one.
        harvest_credentials(cfg, base)
        return ""
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def ensure_config_readable(env: "Mapping[str, str] | None" = None) -> str:
    """Verify the CLI's account file parses; repair it from a known-good copy.

    **A truncated account file costs every launch after it.** If
    `.claude.json` is left mid-write, the CLI exits within seconds of launch
    with "Configuration file ... is corrupted": nothing is attempted and
    nothing is spent, and the caller cannot tell "the solver ran and lost"
    from "the solver never started".

    A game is hours of solver work, and the driver launches solvers
    in-process, so a shell-level check upstream cannot cover each launch. This
    runs immediately before the spawn.

    **Refuses rather than proceeds** when it cannot repair: launching into a
    broken config produces an attempt the driver then discards, which spends the
    slot and banks nothing. A refusal is visible; a discarded attempt looks like
    a hard game.

    The known-good copy is `<account file>.known-good`, refreshed by whoever
    sets a config dir up. Returns ``"ok"`` or ``"restored"``.
    """
    target = config_file(env)
    def _parses(p: Path) -> bool:
        try:
            json.loads(p.read_text())
            return True
        except Exception:                        # noqa: BLE001 -- any failure
            return False

    if target.exists() and _parses(target):
        return "ok"

    backup = target.with_name(target.name + ".known-good")
    if not backup.exists() or not _parses(backup):
        raise RuntimeError(
            f"{target} is missing or unreadable and there is no usable "
            f"{backup.name} to restore from. Refusing to launch: the CLI would "
            f"exit immediately and the attempt would be discarded, spending the "
            f"slot and banking nothing."
        )
    # Keep the damaged file rather than deleting it -- it is the only
    # evidence of how the corruption looked.
    if target.exists():
        os.replace(target, target.with_name(target.name + ".corrupt"))
    shutil.copy2(backup, target)
    os.chmod(target, 0o600)
    if not _parses(target):
        raise RuntimeError(f"restored {target} from {backup.name} and it still "
                           f"does not parse; refusing to launch")
    return "restored"


def config_file(env: "Mapping[str, str] | None" = None) -> Path:
    """The account file that goes with :func:`config_dir`."""
    raw = (env or {}).get("CLAUDE_CONFIG_DIR") or os.environ.get("CLAUDE_CONFIG_DIR")
    return (Path(raw).expanduser().resolve() / ".claude.json") if raw \
        else Path.home() / ".claude.json"

def hidden_paths(workspace: Path, scratch: Path | None = None,
                 env: "Mapping[str, str] | None" = None) -> list[Path]:
    """Every tree the guardrail removes. Existing paths only — see the module note.

    The attempt directory and the solver's project dir are NOT here: they are
    held and re-bound by the script, which is what makes hiding their parents
    safe.
    """
    home = Path.home()
    candidates = [
        _REPO / "tools",
        _REPO / "evidence",
        _REPO / "docs",
        _REPO / "tests",
        _REPO / ".git",
        # Extra trees to hide, given as colon-separated absolute paths in
        # `CCARC3_MASK_DIRS`, so nothing host-specific is written in source.
        *(Path(p) for p in
          os.environ.get("CCARC3_MASK_DIRS", "").split(":") if p),
        config_dir(env) / "projects",
    ]
    if scratch is not None:
        candidates.insert(0, scratch)
    # **A sibling run's workspace is never the agent's to see.**
    #
    # Derived from the workspace, NOT from the `scratch` argument: deriving it
    # means the wall cannot depend on a caller remembering an optional
    # keyword -- an optional keyword is a wall with a gap in it. The sweep
    # directory is the workspace's own third
    # parent (`<sweep>/<game>/attempt_N/<gid>`); hiding it while the attempt dir is
    # held and re-bound leaves the run its own tree and nothing beside it.
    #
    # Guarded, because a workspace that is not in that layout would otherwise hide
    # something far too broad: the candidate must sit under $HOME, and must be
    # neither $HOME nor the repository root.
    parents = Path(workspace).resolve().parents
    if len(parents) > 2:
        sweep = parents[2]
        # A sweep directory lives under $HOME in a real run and under /tmp
        # in a test, so both roots are accepted -- and never a root itself:
        # laying a tmpfs over $HOME or /tmp from here would hide the very
        # trees the script is about to re-bind.
        roots = (home, Path("/tmp"))
        if (any(r in sweep.parents for r in roots)
                and sweep not in (*roots, _REPO)):
            candidates.insert(0, sweep)
    candidates += sorted(Path("/tmp").glob("claude-*"))
    return [p for p in candidates if p.exists()]


# ── strict mode ────────────────────────────────────────────────────────────────
#
# **The enumerated list is a denylist, and a denylist ages badly.** `hidden_paths`
# names specific trees, which is the right shape only while the set of secrets
# is closed: a store nobody thought to enumerate is readable by default.
#
# A denylist cannot be made correct by adding entries: the next store nobody
# enumerated is the next hole. Strict mode inverts it -- lay one tmpfs over the
# home directory and re-bind only what a run genuinely needs, so a store that
# nobody thought of is hidden by construction rather than by an author remembering
# it.
#
# What a run genuinely needs, and nothing else:
#   * the attempt directory -- the workspace and its sidecars, read-write;
#   * the package source -- the solver imports it;
#   * the interpreter -- PATH points into the repo venv;
#   * the CLI's own per-cwd transcript directory -- `--resume` replays it.

def _account_file_if_outside(env: "Mapping[str, str] | None" = None) -> list[Path]:
    """The account file, but only when it is NOT inside the config directory.

    **Keeping it twice truncates it.** With no
    `CLAUDE_CONFIG_DIR` the two are siblings -- `~/.claude` and `~/.claude.json`
    -- so both belong in the keep list. Set the variable and the account file
    moves INSIDE the directory, so the directory's own bind already carries it;
    adding it again makes the guardrail create a second mount point over content it
    has already bound, and the file reads as **0 bytes inside the guardrail** while
    remaining intact on the host.

    A solver launched into that state dies in seconds with "Configuration
    file ... is corrupted: JSON Parse error: Unexpected EOF", with the same
    signature as a hard failure of its own.
    """
    d, f = config_dir(env), config_file(env)
    try:
        f.relative_to(d)
    except ValueError:
        return [f]          # a sibling: keep it explicitly
    return []               # inside the directory: already covered by its bind


def kept_paths(workspace: Path, env: "Mapping[str, str] | None" = None) -> list[Path]:
    """Everything strict mode re-binds. Anything absent from here is gone.

    **The base interpreter is on this list because a venv is a set of symlinks.**
    `.venv/bin/python` here resolves to the distribution under `sys.base_prefix`,
    so laying a tmpfs over `$HOME` without keeping that prefix leaves a dangling
    symlink and `unshare` fails with "No such file or directory" -- at launch,
    after the namespace exists. It is derived from the running interpreter
    rather than written down: a literal interpreter path is correct only on
    the box it was written on.
    """
    import sys as _sys

    # **The repository `src` is deliberately NOT here.** The run imports the
    # staged package, which sits inside the attempt directory and is therefore
    # already kept by the first entry; `src/ccarc` carries the unstripped
    # assets and is not offered to the run.
    # **The solver's own CLI, enumerated because each omission fails at
    # launch, and only after the namespace exists**: the base interpreter (a
    # venv is symlinks), then the `claude` launcher, then the versioned binary
    # it resolves to, then the CLI's config and state. `shutil.which` and
    # `readlink` rather than literals -- a path written down is correct only on
    # the box it was written on.
    cli = shutil.which("claude")
    cli_paths: list[Path] = []
    if cli:
        launcher = Path(cli)
        cli_paths += [launcher.parent, Path(os.path.realpath(cli)).parent]
    home = Path.home()
    out = [Path(workspace).resolve().parent, _REPO / ".venv",
           Path(_sys.base_prefix), Path(_sys.prefix),
           *cli_paths,
           config_dir(env), *_account_file_if_outside(env)]
    seen, keep = set(), []
    for p in out:                      # de-duplicate, keep order
        if p not in seen:
            seen.add(p)
            keep.append(p)
    return keep


_STRICT_SCRIPT = """\
set -e
mount -t tmpfs none /mnt
stage() {
  n=0
  for d in $1; do
    n=$((n+1))
    if [ -d "$d" ]; then mkdir -p "$2/$n"; else mkdir -p "$2"; : > "$2/$n"; fi
    mount --rbind "$d" "$2/$n"
  done
}
rebind() {
  n=0
  for d in $1; do
    n=$((n+1))
    if [ -d "$2/$n" ]; then mkdir -p "$d"; else mkdir -p "$(dirname "$d")"; : > "$d"; fi
    mount --rbind "$2/$n" "$d"
  done
}
mkdir -p /mnt/k /mnt/l
stage "$CCARC3_CONFINE_KEEP" /mnt/k
stage "$CCARC3_CONFINE_KEEP_LAST" /mnt/l
for r in $CCARC3_CONFINE_ROOTS; do
  mount -t tmpfs none "$r"
done
rebind "$CCARC3_CONFINE_KEEP" /mnt/k
for d in $CCARC3_CONFINE_HIDE_AFTER; do
  # Same file-vs-directory rule as the HIDE loop above.
  if [ -e "$d" ] && [ ! -d "$d" ]; then
    : > /mnt/empty
    mount --bind /mnt/empty "$d"
  else
    mkdir -p "$d"
    mount -t tmpfs none "$d"
  fi
done
rebind "$CCARC3_CONFINE_KEEP_LAST" /mnt/l
cd "$CCARC3_CONFINE_CWD"
exec unshare --user --map-user="$CCARC3_CONFINE_UID" \
     --map-group="$CCARC3_CONFINE_GID" -- "$@"
"""


def build_strict_command(workspace: Path, argv: list[str], *,
                         env: dict[str, str] | None = None,
                         extra_keep: list[Path] | None = None) -> list[str]:
    """Wrap ``argv`` so that only :func:`kept_paths` survives under ``$HOME``.

    ``sh -e`` again: a failed mount aborts the launch rather than opening a confinement
    with a wall missing. Paths with whitespace are refused for the same reason
    `build_command` refuses them -- the keep list travels through a shell ``for``.
    """
    workspace = Path(workspace).resolve()
    keep = [p for p in kept_paths(workspace, env) if p.exists()]
    if extra_keep:
        keep += [Path(p).resolve() for p in extra_keep if Path(p).exists()]
    # **Both roots, because both hold secrets.** `$HOME` carries the runs,
    # the holdout and the repository; `/tmp` carries scratch state. Covering
    # only one leaves the other open, and a workspace legitimately lives in
    # either: under `$HOME` in a real run, under `/tmp` in a test.
    roots = [Path.home(), Path("/tmp")]
    for p in [*keep, *roots]:
        if any(c.isspace() for c in str(p)):
            raise ValueError(f"confinement cannot express a path with whitespace: {p}")
    for p in keep:
        if not any(r in p.parents or p == r for r in roots):
            raise ValueError(
                f"strict mode lays a tmpfs over {roots} and re-binds the keep "
                f"list; {p} is outside them and would not survive"
            )
    # **`~/.claude` is kept, `~/.claude/projects` is not.** The CLI needs its own
    # config and auth, which live under `~/.claude`; that directory also holds
    # every other project's transcripts. Keeping the parent and then laying a
    # tmpfs over `projects` -- after the re-bind, before the run's own project
    # directory goes back -- gives the CLI what it needs and no one else's
    # history. The ordering is the whole mechanism: re-binding the run's own
    # directory before the tmpfs would bury it.
    # **Belt and braces, because the key is the CLI's convention, not ours.**
    # It is derived from an undocumented transformation which can change in any
    # CLI release, and the failure mode is silent -- the transcript simply
    # lands on the tmpfs and vanishes. So when the config directory is the
    # run's OWN (per_run_config puts it in the attempt tree), there is nothing
    # to hide: it holds this run's transcripts and no one else's. Keep the
    # whole thing and the transcript survives under whatever key the CLI picks.
    cfg = config_dir(env)
    private = cfg.parent == Path(workspace).resolve().parent
    if private:
        hide_after: list[Path] = []
        keep_last: list[Path] = []
    else:
        hide_after = [cfg / "projects"]
        keep_last = [cfg / "projects" / _project_key(workspace)]
    # Created before staging because the CLI writes it on first use, and a
    # bind-mount source that does not exist aborts the launch. Making it here
    # rather than skipping it keeps `--resume` working: every nudge arm replays
    # that transcript, and a run whose directory was skipped would resume into
    # an empty one. The CLI would create the same path itself.
    for d in keep_last:
        d.mkdir(parents=True, exist_ok=True)
    if env is not None:
        env["CCARC3_CONFINE_CWD"] = str(workspace)
        env["CCARC3_CONFINE_ROOTS"] = " ".join(str(r) for r in roots)
        env["CCARC3_CONFINE_KEEP"] = " ".join(str(p) for p in keep)
        env["CCARC3_CONFINE_HIDE_AFTER"] = " ".join(str(p) for p in hide_after)
        env["CCARC3_CONFINE_KEEP_LAST"] = " ".join(str(p) for p in keep_last)
        env["CCARC3_CONFINE_UID"] = str(os.getuid())
        env["CCARC3_CONFINE_GID"] = str(os.getgid())
    return ["unshare", "--user", "--mount", "--map-root-user",
            "/bin/sh", "-c", _STRICT_SCRIPT, "confine-strict", *argv]


# The outer-namespace script. `sh -e`: any failed mount aborts the launch —
# an incomplete confinement must not open. `$@` is the solver argv, exec'd
# under the inner namespace that restores the real uid.
_SCRIPT = """\
set -e
mount -t tmpfs none /mnt
mkdir -p /mnt/att /mnt/proj
mount --rbind "$CCARC3_CONFINE_ATT" /mnt/att
mkdir -p "$CCARC3_CONFINE_PROJ"
mount --rbind "$CCARC3_CONFINE_PROJ" /mnt/proj
# **A tmpfs cannot be laid over a FILE, and `/tmp/claude-*` matches both.**
# Claude Code leaves directories (`/tmp/claude-<id>/`) AND marker files
# (`/tmp/claude-<id>-cwd`, holding a path). The glob returns both, a tmpfs
# mount fails on a file, and `sh -e` would abort the launch -- one stale marker
# file anywhere in /tmp would make the guardrail refuse to open at all. Failing
# closed is the right direction, but that failure lands on a healthy box.
#
# A file is masked by binding an empty file over it -- same effect, and the
# content is as unreachable as it would be under a tmpfs.
for d in $CCARC3_CONFINE_HIDE; do
  if [ -d "$d" ]; then
    mount -t tmpfs none "$d"
  else
    : > /mnt/empty
    mount --bind /mnt/empty "$d"
  fi
done
mkdir -p "$CCARC3_CONFINE_ATT"
mount --rbind /mnt/att "$CCARC3_CONFINE_ATT"
mkdir -p "$CCARC3_CONFINE_PROJ"
mount --rbind /mnt/proj "$CCARC3_CONFINE_PROJ"
cd "$CCARC3_CONFINE_CWD"
exec unshare --user --map-user="$CCARC3_CONFINE_UID" \
     --map-group="$CCARC3_CONFINE_GID" -- "$@"
"""
# The `cd` above is load-bearing, not tidiness. `Popen(cwd=...)` resolves the
# working directory in the HOST namespace, before any mount lands — so the
# solver would hold a live dentry chain into the *shadowed* original tree, and
# a relative `..` walk from there would reach every sibling the tmpfs was
# supposed to bury, while an absolute path to the same place answers ENOENT.
# Re-entering the path after the mounts resolves it through the new table.


def build_command(workspace: Path, argv: list[str], *,
                  scratch: Path | None = None,
                  env: dict[str, str] | None = None) -> list[str]:
    """Wrap a solver argv in the guardrail. Mutates ``env`` with the script's inputs.

    ``workspace`` is the game workspace (``.../attempt_N/<gid>``); the held
    read-write tree is its PARENT, because the client keeps the session jar
    one level above the workspace by design.

    Paths with whitespace are refused rather than mis-split: the hide list
    travels through a shell ``for`` over an env var, and a path that needs
    quoting would hide the wrong thing silently. Nothing in this project's
    layout has spaces; refusing keeps that an invariant instead of a hope.
    """
    workspace = Path(workspace).resolve()
    attempt = workspace.parent
    hide = hidden_paths(workspace, scratch=scratch, env=env)
    for p in [attempt, *hide]:
        if any(c.isspace() for c in str(p)):
            raise ValueError(f"confinement cannot express a path with "
                             f"whitespace: {p}")
    proj = config_dir(env) / "projects" / _project_key(workspace)
    if env is not None:
        env["CCARC3_CONFINE_CWD"] = str(workspace)
        env["CCARC3_CONFINE_ATT"] = str(attempt)
        env["CCARC3_CONFINE_PROJ"] = str(proj)
        env["CCARC3_CONFINE_HIDE"] = " ".join(str(p) for p in hide)
        env["CCARC3_CONFINE_UID"] = str(os.getuid())
        env["CCARC3_CONFINE_GID"] = str(os.getgid())
    return ["unshare", "--user", "--mount", "--map-root-user",
            "/bin/sh", "-c", _SCRIPT, "confine", *argv]


def describe(workspace: Path, scratch: Path | None = None) -> str:
    """One log line saying what the guardrail hides, so a launch is auditable."""
    hide = hidden_paths(Path(workspace), scratch=scratch)
    return ("confined: rw=" + shlex.quote(str(Path(workspace).parent))
            + " hidden=" + ",".join(p.name for p in hide))
