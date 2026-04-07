# IncidentOps OpenEnv

Deterministic incident-response environment for OpenEnv. Agents must classify severity, gather evidence, escalate correctly, apply mitigations, and resolve incidents within a step budget.

## Environment Overview
- **Domain:** IT / SaaS incident triage and response coordination.
- **Agent Inputs:** alerts, logs, metrics, runbook snippets, impact summaries.
- **Agent Outputs:** structured actions like `query_logs`, `classify_severity`, `apply_mitigation`, `resolve_incident`.

## Action Space
Key action types:
- `query_logs`, `query_metrics`, `read_runbook`, `inspect_dependency`
- `classify_severity`, `escalate_team`, `apply_mitigation`, `post_status_update`, `resolve_incident`

## Observation Space
Includes:
- `incident_id`, `task_id`, `step_index`, `max_steps`
- `title`, `alert_summary`, `customer_impact_summary`
- `visible_logs`, `visible_metrics`, `visible_runbook_notes`
- `actions_taken`, `current_hypotheses`, `available_actions`

## Tasks
- **Easy:** single-service outage, clear rollback/feature flag fix.
- **Medium:** dependency failure with DB saturation; escalation required.
- **Hard:** multi-signal, red herrings, dependency bottleneck; 4-6 step resolution path.

Each task config is deterministic and includes fixed evidence sources.

## Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Server
```
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

## Baseline Inference
```
export API_BASE_URL=...
export MODEL_NAME=...
export HF_TOKEN=...
python inference.py
```

The script prints `[START]`, `[STEP]`, and `[END]` logs for each task using the required format.

## Files
- `app/` environment implementation
- `app/tasks/` deterministic task configs
- `openenv.yaml` OpenEnv spec
- `inference.py` baseline

## Validation Checklist
- `openenv validate` passes
- Docker builds and runs
- HF Space responds to `/reset`
- Baseline scores are reproducible

## Local Pre-Submission Validation
```
python scripts/validate.py
```

## Submission Validator Script
```
chmod +x scripts/validate-submission.sh
./scripts/validate-submission.sh https://your-space.hf.space /Users/satyambirla/Desktop/Meta/IncidentTriage-main
```
