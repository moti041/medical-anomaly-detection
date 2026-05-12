"""
Integration with ml_flow_like experiment tracker.
Provides utilities to log runs, parameters, metrics, and artifacts.
"""

import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add ml_flow_like to path
ML_FLOW_PATH = Path("C:\\Users\\Public\\ml_flow_like")
sys.path.insert(0, str(ML_FLOW_PATH))

try:
    from backend.db import get_connection
    from backend.repository import add_artifact, create_run, get_run, run_exists
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    print("[Warning] ml_flow_like tracker not available")


def get_git_commit_hash():
    try:
        output = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path.cwd(), text=True)
        return output.strip()
    except Exception:
        return None


def _merge_existing_run(run_id, experiment_name, parameters, metrics, notes, tags):
    existing = get_run(run_id) or {}
    merged_parameters = {**existing.get("parameters", {}), **parameters}
    merged_metrics = {**existing.get("metrics", {}), **metrics}
    merged_notes = notes or existing.get("notes", "")

    import json

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE runs
            SET experiment_name = ?, parameters_json = ?, metrics_json = ?, notes = ?
            WHERE run_id = ?
            """,
            (
                experiment_name,
                json.dumps(merged_parameters),
                json.dumps(merged_metrics),
                merged_notes,
                run_id,
            ),
        )
        for tag in sorted({tag.strip() for tag in tags if tag.strip()}):
            conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
            conn.execute(
                """
                INSERT OR IGNORE INTO run_tags (run_id, tag_id)
                VALUES (?, (SELECT id FROM tags WHERE name = ?))
                """,
                (run_id, tag),
            )
    return get_run(run_id)


def log_run(
    experiment_name: str,
    parameters: Dict[str, Any],
    metrics: Dict[str, Any],
    notes: str = "",
    tags: List[str] = None,
    artifacts: List[str] = None,
) -> Optional[str]:
    """
    Log an experiment run to the tracker.

    Args:
        experiment_name: Name of the experiment (e.g., "chest_xray_anomaly_detection")
        parameters: Dict of hyperparameters and configuration
        metrics: Dict of evaluation metrics (accuracy, loss, etc)
        notes: Optional description of the run
        tags: Optional list of tags (e.g., ["baseline", "improved", "v2"])
        artifacts: Optional list of file paths to save as artifacts

    Returns:
        run_id if successful, or fallback local run_id if tracker unavailable
    """
    run_id = parameters.get('run_id') or str(uuid.uuid4())
    parameters['git_commit_hash'] = parameters.get('git_commit_hash') or get_git_commit_hash()

    if not TRACKER_AVAILABLE:
        print(f"[Warning] Tracker unavailable; generated local run_id {run_id}")
        return run_id

    try:
        if run_exists(run_id):
            _merge_existing_run(run_id, experiment_name, parameters, metrics, notes, tags or [])
        else:
            create_run(
                run_id=run_id,
                experiment_name=experiment_name,
                parameters=parameters,
                metrics=metrics,
                notes=notes,
                tags=tags or [],
            )
        print(f"[Tracker] Run logged: {run_id}")

        if artifacts:
            for artifact_path in artifacts:
                artifact_file = Path(artifact_path)
                if artifact_file.exists():
                    try:
                        add_artifact(run_id, artifact_file.name, artifact_file.resolve())
                        print(f"[Tracker] Artifact saved: {artifact_file.name}")
                    except Exception as e:
                        print(f"[Warning] Failed to save artifact {artifact_file.name}: {e}")
                else:
                    print(f"[Warning] Artifact missing: {artifact_path}")

        return run_id

    except Exception as e:
        print(f"[Error] Failed to log run: {e}")
        return run_id
