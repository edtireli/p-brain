import os
import subprocess


def get_git_version() -> str:
    """Return version from git tags."""
    try:
        # --tags ensures annotated and lightweight tags are considered
        version = subprocess.check_output(
            ["git", "describe", "--tags"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        version = "0.0.0"
    return version


__version__ = get_git_version()
