"""
Top-level run script for Code Ocean capsule.

For local development:
    panel serve code/app.py --dev --show

For production (Code Ocean):
    python code/run_capsule.py
"""

import subprocess
import sys


def run():
    """Start the Panel server."""
    cmd = [
        sys.executable, "-m", "panel", "serve",
        "code/app.py",
        "--address", "0.0.0.0",
        "--port", "7860",
        "--allow-websocket-origin=*",
        "--dev",
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    run()