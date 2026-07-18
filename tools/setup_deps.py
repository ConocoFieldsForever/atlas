"""Python dependency setup for the build pipeline, run from the menu's INSTALL DEPS button.

Streams into the menu's build panel (no app restart needed). Creates a venv beside the kit and
installs the base extraction requirements (UnityPy, numpy, Pillow) into it; the viewer's
paths::python_exe() then prefers that venv automatically on the next spawn. ASCII output only, with
[STAGE i/N] / [BUILD OK] / [BUILD FAILED] markers so the menu's loading bar + outcome logic work.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)  # repo/bundle root (holds venv/ and extraction/)
VENV = os.path.join(KIT, "venv")
REQ = os.path.join(KIT, "extraction", "requirements.txt")
TOTAL = 3


def vpy():
    sub, exe = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return os.path.join(VENV, sub, exe)


def run(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="ascii", errors="replace",
    )
    for line in p.stdout:
        print("  " + line.rstrip(), flush=True)
    return p.wait()


def main():
    print("[SETUP] installing the Python packages the build pipeline needs (UnityPy, numpy, Pillow)",
          flush=True)

    print(f"[STAGE 1/{TOTAL}] create virtual environment", flush=True)
    if not os.path.isfile(vpy()):
        rc = run([sys.executable, "-m", "venv", VENV])
        if rc != 0 or not os.path.isfile(vpy()):
            print(f"[BUILD FAILED] could not create venv at {VENV} (rc={rc})", flush=True)
            sys.exit(2)
    print(f"[STAGE 1/{TOTAL}] create virtual environment: done", flush=True)
    py = vpy()

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
