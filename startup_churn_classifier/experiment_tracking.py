from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from startup_churn_classifier.config import ARTIFACTS_DIR, RESULTS_DIR


RUNS_DIR = RESULTS_DIR / "runs"
EXPERIMENT_LOG_PATH = RESULTS_DIR / "experiments.jsonl"
LATEST_RUN_PATH = RESULTS_DIR / "latest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_manifest(artifact_dir: Path = ARTIFACTS_DIR) -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {}
    if not artifact_dir.exists():
        return manifest

    for path in sorted(artifact_dir.iterdir()):
        if not path.is_file():
            continue
        manifest[path.name] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return manifest


def _artifact_version(manifest: dict[str, dict[str, object]]) -> str:
    payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def log_experiment_run(
    *,
    summary: dict[str, object],
    hyperparameters: dict[str, object],
    artifact_dir: Path = ARTIFACTS_DIR,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    manifest = build_artifact_manifest(artifact_dir)
    artifact_version = _artifact_version(manifest)

    record = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "dataset_path": summary["dataset_path"],
        "rows": summary["rows"],
        "positive_rate": summary["positive_rate"],
        "selected_model": summary["selected_model"],
        "hyperparameters": hyperparameters,
        "metrics": summary["results"],
        "artifact_version": artifact_version,
        "artifacts": manifest,
    }

    run_path = RUNS_DIR / f"{run_id}.json"
    run_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    with EXPERIMENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    LATEST_RUN_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "artifact_version": artifact_version,
        "run_path": str(run_path),
        "latest_path": str(LATEST_RUN_PATH),
    }
