from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(command: list[str]) -> int:
    print(f"$ {' '.join(command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


def install() -> int:
    return run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def train() -> int:
    return run_command([sys.executable, "train.py"])


def serve() -> int:
    return run_command(
        [sys.executable, "-m", "uvicorn", "startup_churn_classifier.api.main:app", "--reload"]
    )


def test() -> int:
    return run_command([sys.executable, "-m", "pytest"])


def docker_build() -> int:
    return run_command(["docker", "build", "-t", "startup-churn-classifier", "."])


def docker_run() -> int:
    return run_command(["docker", "run", "-p", "8000:8000", "startup-churn-classifier"])


def clean() -> int:
    for path in [
        PROJECT_ROOT / "artifacts",
        PROJECT_ROOT / ".pytest_cache",
        PROJECT_ROOT / "results" / "runs",
    ]:
        if path.exists():
            shutil.rmtree(path)

    for path in [
        PROJECT_ROOT / "results" / "experiments.jsonl",
        PROJECT_ROOT / "results" / "latest.json",
        PROJECT_ROOT / "data" / "raw" / "startup_churn.csv",
    ]:
        if path.exists():
            path.unlink()

    print("Removed generated artifacts, cached test files, tracked experiment outputs, and synthetic data.")
    return 0


TASKS = {
    "install": install,
    "train": train,
    "serve": serve,
    "test": test,
    "docker-build": docker_build,
    "docker-run": docker_run,
    "clean": clean,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task runner for common project workflows.")
    parser.add_argument("task", choices=sorted(TASKS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return TASKS[args.task]()


if __name__ == "__main__":
    raise SystemExit(main())
