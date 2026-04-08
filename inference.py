from __future__ import annotations

import json
import os
from typing import List, Optional

from openai import OpenAI

from server.env import IncidentOpsEnv
from server.graders import grade_episode

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = os.getenv("INCIDENT_OPS_BENCHMARK", "incident_ops_env")
MAX_STEPS = 16
SUCCESS_SCORE_THRESHOLD = 0.1


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.4f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


def get_client() -> OpenAI:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN must be set for OpenAI client auth")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def choose_action(client: OpenAI, observation: dict) -> dict:
    system = (
        """You are an incident response agent. Output ONLY valid JSON for the next action.

You are given the current observation of a production incident.
Your job is to choose the NEXT BEST ACTION.

You MUST output EXACTLY ONE valid JSON object representing an action.

-------------------------------------
⚠️ STRICT OUTPUT RULES (VERY IMPORTANT)
-------------------------------------

- Output ONLY JSON (no explanation, no text)
- Use EXACT field names from schema
- Do NOT invent fields
- Do NOT rename fields
- JSON must be valid and parseable

-------------------------------------
✅ AVAILABLE ACTIONS AND SCHEMA
-------------------------------------

1. Query Metrics:
{
  "action_type": "query_metrics",
  "service": "<string>",
  "metric_name": "<string>",
  "window_min": <integer between 1 and 240>
}

2. Query Logs:
{
  "action_type": "query_logs",
  "service": "<string>",
  "query": "<string>",
  "window_min": <integer between 1 and 240>
}

3. Read Runbook:
{
  "action_type": "read_runbook",
  "service": "<string>",
  "topic": "<string>"
}

4. Inspect Dependency:
{
  "action_type": "inspect_dependency",
  "source_service": "<string>",
  "target_service": "<string>"
}

5. Classify Severity:
{
  "action_type": "classify_severity",
  "severity": "sev1" | "sev2" | "sev3",
  "justification": "<string>"
}

6. Escalate Team:
{
  "action_type": "escalate_team",
  "team": "backend" | "database" | "infra" | "payments" | "frontend" | "sre",
  "urgency": "low" | "medium" | "high",
  "justification": "<string>"
}

7. Apply Mitigation:
{
  "action_type": "apply_mitigation",
  "mitigation": "restart_service" | "rollback_deploy" | "shift_traffic" | "clear_queue" | "scale_up" | "toggle_feature_flag",
  "target_service": "<string>",
  "justification": "<string>"
}

8. Post Status Update:
{
  "action_type": "post_status_update",
  "audience": "internal" | "customer",
  "message": "<string>"
}

9. Resolve Incident:
{
  "action_type": "resolve_incident",
  "suspected_root_cause": "<string>",
  "mitigation_applied": "<string>",
  "customer_impact_status": "resolved" | "mitigated" | "unknown",
  "resolution_summary": "<string>",
  "confidence": <float between 0 and 1>
}

-------------------------------------
🚫 COMMON MISTAKES (DO NOT DO)
-------------------------------------

- DO NOT use "metrics" → use "metric_name"
- DO NOT use "duration" or "duration_minutes" → use "window_min"
- DO NOT use "time_range"
- DO NOT use "search_term" → use "query"
- DO NOT output multiple actions
- DO NOT add extra fields

-------------------------------------
🧠 DECISION GUIDELINES
-------------------------------------

- Start by gathering evidence (metrics/logs)
- Use query_metrics for performance issues
- Use query_logs for errors/exceptions
- Use inspect_dependency if issue may be downstream
- Use read_runbook if unsure
- Classify severity when confident
- Apply mitigation only when root cause is likely
- Escalate if needed
- Resolve ONLY when confident

-------------------------------------
📥 INPUT (OBSERVATION)
-------------------------------------

{observation}

-------------------------------------
📤 OUTPUT
-------------------------------------

Return ONLY the JSON action."""
    )
    user = {
        "observation": observation,
        "allowed_action_types": observation.get("available_actions", []),
        "instruction": "Return the next action JSON only.",
    }

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        temperature=0.0,
        max_tokens=400,
    )

    text = (response.choices[0].message.content or "{}").strip()
    
    # Qwen and other Instruct models often output Markdown code blocks.
    # We must strip them out to get valid JSON.
    import re
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1).strip()
        
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    return {
        "action_type": "post_status_update",
        "audience": "internal",
        "message": f"invalid action payload: {text[:50]}...",
    }


def run_task(task_id: str, client: OpenAI) -> None:
    env = IncidentOpsEnv(task_id=task_id)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id=task_id)
        max_steps = min(MAX_STEPS, obs.max_steps)
        for step in range(1, max_steps + 1):
            action = choose_action(client, obs.model_dump())
            result = env.step(action)
            reward = float(result.reward.value)
            done = bool(result.done)
            error = result.info.get("last_action_error")

            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps(action, separators=(",", ":"))
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            obs = result.observation
            if done:
                score = float(result.info.get("final_score", 0.0))
                success = score >= SUCCESS_SCORE_THRESHOLD
                break
        if not success and score == 0.0:
            score = grade_episode(env.state())
            success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = get_client()
    for task_id in ["easy", "medium", "hard"]:
        run_task(task_id, client)


if __name__ == "__main__":
    main()