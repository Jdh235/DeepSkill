"""Opt-in, zero-overhead-when-off debug logging for tracking down the
candidate-sweep job's unexplained mass-SIGKILL.

Every recorded incident died with bigcodebench's evaluate() at 0-2%
progress, and no visible traceback anywhere -- because stdout on this
path is reserved for JSON returned to the parent process (breaking it
would corrupt the parent's json.loads()), nothing has ever been directly
observed about what's happening *inside* a per-instance subprocess at
the moment of death. This writes timestamped, immediately-flushed lines
to a separate file instead, so even a SIGKILL mid-write leaves a partial
trail up to the last completed write -- enough to see exactly which
stage (code_extract's O(n^2) parse loop vs. the sandboxed
multiprocessing.Process execution) the last few instances were in.

Set BIGCODEBENCH_DEBUG_LOG=/path/to/file to enable; unset (the default)
is a true no-op, so this has zero effect on normal runs.
"""

import os
import time

_LOG_PATH = os.environ.get("BIGCODEBENCH_DEBUG_LOG")


def dbg(msg: str) -> None:
    if not _LOG_PATH:
        return
    # os.getpid() must be read fresh on every call, not cached at import
    # time: a forked child inherits the parent's already-imported module
    # state (including any module-level variable computed once at
    # import), so a cached PID would silently show the parent's pid in
    # every line the child itself logs.
    line = f"{time.time():.3f} pid={os.getpid()} {msg}\n"
    try:
        with open(_LOG_PATH, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass
