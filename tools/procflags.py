"""Subprocess creation flags that keep a build from stealing focus from the game.

THE PROBLEM, precisely. The menu spawns `build_map.py` with `DETACHED_PROCESS | CREATE_NO_WINDOW`
when "Process in background" is on (`viewer/src/menu.rs:104`, via `prep_child(cmd, true)`).
DETACHED_PROCESS means the child does not inherit the parent's console and is not given one, so
`build_map.py` runs with NO console at all.

A process with no console that spawns a CONSOLE application without any creation flags makes Windows
allocate a brand-new console *with a visible window* for that child. That window appears on top and
takes foreground. During a build that is dozens of them -- every extractor, every pip call, every
per-level worker -- each one yanking focus out of a live raid.

CREATE_NO_WINDOW gives the child a console with no window, which is what the top of the chain
already asks for. It has to be repeated at every spawn because the flag is not inherited: the
parent's "no window" state does not propagate, only an actual console handle would, and
DETACHED_PROCESS is precisely the case where there is none to inherit.

Applies to console apps. `atlas.exe` is a GUI-subsystem binary and never shows a console either way,
but the flag is harmless there and passing it uniformly means no one has to remember which is which.
"""

import subprocess
import sys

# winbase.h. Not inherited by children; must be passed per spawn.
CREATE_NO_WINDOW = 0x0800_0000


def no_window(**kwargs):
    """Merge CREATE_NO_WINDOW into subprocess kwargs on Windows; a no-op elsewhere.

    Use as `subprocess.Popen(cmd, **no_window(stdout=..., cwd=...))`. On POSIX a non-zero
    `creationflags` raises, so it is omitted rather than zeroed.
    """
    if sys.platform == "win32":
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | CREATE_NO_WINDOW
    return kwargs


def popen(cmd, **kwargs):
    """`subprocess.Popen` that never pops a console window."""
    return subprocess.Popen(cmd, **no_window(**kwargs))


def run(cmd, **kwargs):
    """`subprocess.run` that never pops a console window."""
    return subprocess.run(cmd, **no_window(**kwargs))


def check_output(cmd, **kwargs):
    """`subprocess.check_output` that never pops a console window."""
    return subprocess.check_output(cmd, **no_window(**kwargs))
