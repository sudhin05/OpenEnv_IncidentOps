from __future__ import annotations

import json
from pathlib import Path

from app.env import IncidentOpsEnv
from app.models import TaskConfig

TASK_IDS = ["easy", "medium", "hard"]


def load_task_config(task_id: str) -> TaskConfig:
    tasks_dir = Path(__file__).resolve().parent / "app" / "tasks"
    path = tasks_dir / f"{task_id}_incident.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing task config: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TaskConfig.model_validate(raw)


def assert_required_evidence(config: TaskConfig) -> list[str]:
    errors: list[str] = []
    all_keys = {
        entry.evidence_key
        for entry in config.evidence.logs
        + config.evidence.metrics
        + config.evidence.runbooks
    }

    for key in config.required_evidence_keys:
        if key not in all_keys:
            errors.append(f"Task {config.task_id}: missing required evidence key {key}")

    return errors


def assert_initial_visibility(config: TaskConfig) -> list[str]:
    errors: list[str] = []

    log_keys = {entry.evidence_key for entry in config.evidence.logs}
    metric_keys = {entry.evidence_key for entry in config.evidence.metrics}
    runbook_keys = {entry.evidence_key for entry in config.evidence.runbooks}

    for key in config.initial_visible.log_keys:
        if key not in log_keys:
            errors.append(f"Task {config.task_id}: initial log key {key} not found")

    for key in config.initial_visible.metric_keys:
        if key not in metric_keys:
            errors.append(f"Task {config.task_id}: initial metric key {key} not found")

    for key in config.initial_visible.runbook_keys:
        if key not in runbook_keys:
            errors.append(f"Task {config.task_id}: initial runbook key {key} not found")

    return errors


def assert_seed_and_budget(config: TaskConfig) -> list[str]:
    errors: list[str] = []
    if config.seed < 0:
        errors.append(f"Task {config.task_id}: seed must be non-negative")
    if config.max_steps < 3:
        errors.append(f"Task {config.task_id}: max_steps too small")
    return errors


def validate_task_configs() -> list[str]:
    errors: list[str] = []
    for task_id in TASK_IDS:
        config = load_task_config(task_id)
        errors.extend(assert_required_evidence(config))
        errors.extend(assert_initial_visibility(config))
        errors.extend(assert_seed_and_budget(config))
    return errors


def validate_required_files() -> list[str]:
    errors: list[str] = []
    root = Path(__file__).resolve().parent
    required = [
        root / "openenv.yaml",
        root / "inference.py",
        root / "Dockerfile",
        root / "requirements.txt",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"Missing required file: {path}")
    return errors


def smoke_test_env() -> list[str]:
    errors: list[str] = []
    for task_id in TASK_IDS:
        env = IncidentOpsEnv(task_id=task_id)
        try:
            obs = env.reset(task_id=task_id)
            if obs.task_id != task_id:
                errors.append(f"Env reset mismatch for {task_id}")
        finally:
            env.close()
    return errors


def main() -> None:
    errors: list[str] = []
    errors.extend(validate_required_files())
    errors.extend(validate_task_configs())
    errors.extend(smoke_test_env())

    if errors:
        print("VALIDATION FAILED")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("VALIDATION OK")


if __name__ == "__main__":
    main()