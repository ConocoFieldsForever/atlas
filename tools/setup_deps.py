"""Python dependency setup for the build pipeline, run from the menu's INSTALL DEPS button.

Streams into the menu's build panel (no app restart needed). Installs the base extraction
requirements (UnityPy, numpy, Pillow) into the Python the build pipeline will use, then the viewer's
paths::python_exe() finds them on the next spawn. Two cases:
  * BUNDLED embeddable Python (shipped in the -Full release, `<kit>/python/python.exe`): pip-install
    DIRECTLY into it. The embeddable dist has no working venv/ensurepip, and it is already an
    isolated per-kit interpreter, so a venv is neither possible nor needed.
  * SYSTEM Python (dev / user-installed): create an isolated venv beside the kit and install there,
    so we never pollute the user's global site-packages.
The app runs THIS script with paths::python_exe(), so when a bundled Python exists we are already
executing under it (sys.executable == the bundled python). ASCII output only, with [STAGE i/N] /
[BUILD OK] / [BUILD FAILED] markers so the menu's loading bar + outcome logic work.
"""

import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)  # repo/bundle root (holds python/ or venv/, and extraction/)
VENV = os.path.join(KIT, "venv")
REQ = os.path.join(KIT, "extraction", "requirements.txt")
TOTAL = 3


def vpy():
    sub, exe = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return os.path.join(VENV, sub, exe)


def is_embeddable(exe):
    """A Windows embeddable Python is marked by a pythonNN._pth file beside python.exe."""
    return bool(glob.glob(os.path.join(os.path.dirname(os.path.abspath(exe)), "python*._pth")))


def run(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="ascii", errors="replace",
    )
    for line in p.stdout:
        print("  " + line.rstrip(), flush=True)
    return p.wait()


def has_pip(py):
    try:
        return subprocess.run(
            [py, "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
    except OSError:
        return False


def main():
    print("[SETUP] installing the Python packages the build pipeline needs (UnityPy, numpy, Pillow)",
          flush=True)

    if is_embeddable(sys.executable):
        # Bundled embeddable Python: install straight into it (no venv possible or needed).
        print(f"[STAGE 1/{TOTAL}] use bundled Python (no venv needed)", flush=True)
        py = sys.executable
        print(f"[STAGE 1/{TOTAL}] use bundled Python: done", flush=True)
    else:
        print(f"[STAGE 1/{TOTAL}] create virtual environment", flush=True)
        if not os.path.isfile(vpy()):
            rc = run([sys.executable, "-m", "venv", VENV])
            if rc != 0 or not os.path.isfile(vpy()):
                print(f"[BUILD FAILED] could not create venv at {VENV} (rc={rc})", flush=True)
                sys.exit(2)
        py = vpy()
        if not has_pip(py):
            # Debian/Ubuntu strip the bundled pip wheel out of ensurepip, so a "successful" venv
            # (python -m venv exits 0, the python binary is there) can still come up with NO pip
            # (python -m pip then fails with "No module named pip"). Recreate in place with
            # --system-site-packages so it sees the distro's python3-pip instead of needing a
            # bundled wheel; venv supports re-running on an existing dir without --clear.
            print(
                "  venv has no pip (Debian/Ubuntu strip the bundled wheel) -"
                " retrying with --system-site-packages",
                flush=True,
            )
            run([sys.executable, "-m", "venv", "--system-site-packages", VENV])
            if not has_pip(py):
                print(
                    "[BUILD FAILED] this Python has no usable pip, even system-wide. Install it"
                    " (e.g. `sudo apt install python3-pip`) and try INSTALL DEPS again.",
                    flush=True,
                )
                sys.exit(2)
        print(f"[STAGE 1/{TOTAL}] create virtual environment: done", flush=True)

    print(f"[STAGE 2/{TOTAL}] install packages (downloads from PyPI - needs internet)", flush=True)
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    if os.path.isfile(REQ):
        rc = run([py, "-m", "pip", "install", "-r", REQ])
    else:
        rc = run([py, "-m", "pip", "install", "UnityPy==1.25.0", "numpy>=1.26", "Pillow>=10.0"])
    if rc != 0:
        print(f"[BUILD FAILED] pip install failed (rc={rc}) - check your internet connection", flush=True)
        sys.exit(rc or 1)
    print(f"[STAGE 2/{TOTAL}] install packages: done", flush=True)

    print(f"[STAGE 3/{TOTAL}] verify", flush=True)
    if run([py, "-c", "import UnityPy, numpy, PIL; print('deps OK')"]) != 0:
        print("[BUILD FAILED] packages still missing after install", flush=True)
        sys.exit(3)
    print(f"[STAGE 3/{TOTAL}] verify: done", flush=True)
    print("[BUILD OK] dependencies installed - you can BUILD maps now", flush=True)


if __name__ == "__main__":
    main()
