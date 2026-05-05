"""
Synthetic Data Generation Pipeline — One-click Start Script

Usage:
    python start.py              (default: port 8080)
    python start.py --port 9000  (custom port)
"""

import subprocess
import sys
import os
import shutil

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PORT = 8080


def run(cmd, description, check=True):
    """Run a shell command with a description."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  > {cmd}\n")
    result = subprocess.run(cmd, shell=True, cwd=PROJECT_DIR)
    if check and result.returncode != 0:
        print(f"\n[ERROR] {description} failed (exit code {result.returncode})")
        sys.exit(1)
    return result


def check_uv():
    """Check if uv is installed, guide user if not."""
    if shutil.which("uv"):
        return True
    print("\n[ERROR] 'uv' is not installed.")
    print("  Install it with:  pip install uv")
    print("  Or see: https://docs.astral.sh/uv/getting-started/installation/")
    sys.exit(1)


def ensure_env_file():
    """Check that .env file exists."""
    env_path = os.path.join(PROJECT_DIR, ".env")
    if not os.path.exists(env_path):
        print("\n[WARNING] .env file not found!")
        print("  Create a .env file with your Azure OpenAI credentials.")
        print("  See .env.example for reference.")
        sys.exit(1)


def ensure_directories():
    """Create required directories if they don't exist."""
    for d in ["data", "cache", "output"]:
        path = os.path.join(PROJECT_DIR, d)
        os.makedirs(path, exist_ok=True)


def parse_port():
    """Parse --port from command line args."""
    port = DEFAULT_PORT
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            try:
                port = int(args[i + 1])
            except ValueError:
                print(f"[ERROR] Invalid port: {args[i + 1]}")
                sys.exit(1)
        elif arg.startswith("--port="):
            try:
                port = int(arg.split("=", 1)[1])
            except ValueError:
                print(f"[ERROR] Invalid port: {arg}")
                sys.exit(1)
    return port


def main():
    port = parse_port()

    print(r"""
    ╔══════════════════════════════════════════════════════════╗
    ║   Synthetic Data Generation Pipeline                    ║
    ║   Starting up...                                        ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Step 1: Check prerequisites
    check_uv()
    ensure_env_file()
    ensure_directories()

    # Step 2: Install / sync dependencies
    run("uv sync", "Installing dependencies")

    # Step 3: Quick import check
    result = run(
        'uv run python -c "from app.agents.supervisor import Supervisor; print(\'All modules OK\')"',
        "Verifying project imports",
        check=False,
    )
    if result.returncode != 0:
        print("\n[ERROR] Import check failed. Check the error above.")
        sys.exit(1)

    # Step 4: Start the server
    print(f"\n{'='*60}")
    print(f"  Starting server on http://localhost:{port}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    try:
        subprocess.run(
            f"uv run python -m uvicorn app.main:app --host 0.0.0.0 --port {port} --reload",
            shell=True,
            cwd=PROJECT_DIR,
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped.")


if __name__ == "__main__":
    main()
