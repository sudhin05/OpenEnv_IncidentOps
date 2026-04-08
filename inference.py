from __future__ import annotations

import json
import os
from typing import List, Optional

from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

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
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def get_client() -> OpenAI:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN must be set for OpenAI client auth")
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def choose_action(client: OpenAI, observation: dict) -> dict:
    action_definitions = """
Available Actions and required fields:
- query_logs(service: str, query: str, window_min: int)
- query_metrics(service: str, metric_name: str, window_min: int)
- read_runbook(service: str, topic: str)
- inspect_dependency(source_service: str, target_service: str)
- classify_severity(severity: "sev1" | "sev2" | "sev3", justification: str)
- escalate_team(team: "backend" | "database" | "infra" | "payments" | "frontend" | "sre", urgency: "low" | "medium" | "high", justification: str)
- apply_mitigation(mitigation: "restart_service" | "rollback_deploy" | "shift_traffic" | "clear_queue" | "scale_up" | "toggle_feature_flag", target_service: str, justification: str)
- post_status_update(audience: "internal" | "customer", message: str)
- resolve_incident(suspected_root_cause: str, mitigation_applied: str, customer_impact_status: "resolved" | "mitigated" | "unknown", resolution_summary: str, confidence: float)

All actions MUST include "action_type" field matching the name above.
"""
    system = (
        "You are an expert SRE. Be methodical and thorough.\n"
        "1. INVESTIGATE: Query for EVERY service mentioned in the alert. Use keywords: 'error', 'latency', 'db'.\n"
        "2. CHAIN: When a log mentions a downstream service, query THAT service next.\n"
        "3. EVIDENCE QUOTA: You MUST have 3+ distinct logs/metrics in your history. If you have fewer, keep searching. Resolution with low evidence will be REJECTED.\n"
        "4. RESOLVE: Only after finding the cause AND applying a mitigation, call 'resolve_incident'.\n"
        f"{action_definitions}"
    )
    
    findings_count = len(observation.get("visible_logs", [])) + len(observation.get("visible_metrics", []))
    user = {
        "observation": observation,
        "instruction": (
            f"Findings: {findings_count}/3 required. Step: {observation.get('step_index')}/{observation.get('max_steps')}. "
            "If findings < 3, you MUST use query_logs or query_metrics now. Use simple keywords."
        )
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
    
    # Handle Markdown JSON blocks
    if text.startswith("```"):
        lines = text.splitlines()
        # Find the first line that looks like meat
        start = 1 if lines[0].startswith("```") else 0
        end = -1 if lines[-1].startswith("```") else len(lines)
        text = "\n".join(lines[start:end])

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        # Fallback extraction
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group())
                if isinstance(payload, dict): return payload
            except: pass

    return {
        "action_type": "post_status_update",
        "audience": "internal",
        "message": "Auto-recovery: Agent produced invalid JSON or failed to follow format.",
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
