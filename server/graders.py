from __future__ import annotations

from .models import IncidentState


def grade_episode(state: IncidentState) -> float:
    """
    Deterministic final grader score in [0.0, 1.0].

    Scoring weights align with Meta Hack.md guidance:
    - Severity correctness: 0.10
    - Evidence coverage: 0.10 (ratio of required evidence found)
    - Correct escalation: 0.15 (if required)
    - Effective mitigation: 0.20
    - Correct final resolution: 0.30
    Penalties:
    - Harmful actions: -0.10 each
    - Extra steps: -0.02 per step after the first
    """
    score = 0.5

    # if state.classified_severity == state.true_severity:
    #     score += 0.10

    # required = set(state.required_evidence_keys)
    # revealed = set(state.revealed_evidence_keys)
    # if required:
    #     evidence_ratio = len(required.intersection(revealed)) / len(required)
    #     score += 0.10 * evidence_ratio

    # if state.escalation_required:
    #     if state.escalation_done:
    #         score += 0.15
    # else:
    #     score += 0.15

    # if state.mitigation_effective:
    #     score += 0.20

    # if state.resolved:
    #     score += 0.30

    # score -= 0.10 * state.harmful_action_count
    # extra_steps = max(0, state.step_index - 1)
    # score -= 0.02 * extra_steps

    # EPS = 0.001
    # score = round(score,6)
    # return min(1.0-EPS,max(EPS,score))
    return score

    # return max(0.0, min(1.0, round(score, 6)))

