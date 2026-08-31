"""Provenance helpers, ported from csrc/harness/provenance.h.

The source revision is marked -dirty when anything that can change a result
differs from HEAD: kernels and harnesses (C++ and Python), scripts, build
rules and the pinned Python environment.
"""
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRACKED = ["csrc", "triton_kernels", "scripts", "Makefile",
           "requirements.txt", "requirements-lock.txt"]


def _git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def source_revision():
    rev = _git("rev-parse", "--short=12", "HEAD")
    if not rev:
        return "unknown"
    # __pycache__ directories are build products, not source changes.
    status = [line for line in _git("status", "--porcelain", "--", *TRACKED).splitlines()
              if "__pycache__" not in line]
    return rev + "-dirty" if status else rev


def is_untraceable(rev):
    return rev == "unknown" or rev.endswith("-dirty")


def utc_timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def gpu_selector(props):
    """nvidia-smi --id value for the device torch runs on."""
    uuid = str(getattr(props, "uuid", "") or "")
    if not uuid:
        return ""
    return uuid if uuid.startswith("GPU-") else "GPU-" + uuid


def valid_device_tag(tag):
    return bool(re.fullmatch(r"[a-z0-9-]+", tag or ""))


def csv_safe(s):
    return str(s).replace(",", ";")


def fmt_int(v):
    return "" if v is None or v < 0 else str(int(v))


def fmt_num(v, spec):
    if v is None or v != v or v < 0 or v == float("inf"):
        return ""
    return format(v, spec)
