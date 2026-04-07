from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from .graders import grade_episode
from .models import (
    ApplyMitigationAction,
    ClassifySeverityAction,
    EscalateTeamAction,
    IncidentAction,
    IncidentObservation,
    IncidentState,
    InspectDependencyAction,
    LogSnippet,
    MetricSnapshot,
    PostStatusUpdateAction,
    QueryLogsAction,
    QueryMetricsAction,
    ReadRunbookAction,
    ResolveIncidentAction,
    StepResult,
    StepReward,
    TaskConfig,
)


class IncidentOpsEnv:
    """
    IncidentOps: Causal Incident Command Under Uncertainty

    A deterministic, seeded environment for incident investigation and response.
    The agent interacts through typed actions and receives compact observations.
    """

    AVAILABLE_ACTIONS = [
        "query_logs",
        "query_metrics",
        "read_runbook",
        "inspect_dependency",
        "classify_severity",
        "escalate_team",
        "apply_mitigation",
        "post_status_update",
        "resolve_incident",
    ]

    def __init__(self, task_id: str = "easy") -> None:
        self.task_id = task_id
        self._state: IncidentState | None = None
        self._milestones_awarded: set[str] = set()

    def reset(self, task_id: str | None = None) -> IncidentObservation:
        if task_id is not None:
            self.task_id = task_id

        config = self._load_task_config(self.task_id)
        self._state = self._build_state_from_config(config)
        self._milestones_awarded = set()
        return self._build_observation()

    def state(self) -> IncidentState:
        self._require_state()
        return self._state.model_copy(deep=True)

    def close(self) -> None:
        self._state = None
        self._milestones_awarded = set()

    def step(self, action: IncidentAction | dict) -> StepResult:
        self._require_state()

        parsed_action, invalid_reason = self._parse_action(action)
        state = self._state

        if state.resolved:
            reward = StepReward(
                value=0.0,
                components={"already_done": 0.0},
            )
            return StepResult(
                observation=self._build_observation(),
                reward=reward,
                done=True,
                info={"reason": "incident_already_resolved"},
            )

        state.step_index += 1
        reward_components: dict[str, float] = {}

        if state.step_index > 1:
            reward_components["step_penalty"] = -0.02

        last_action_error: str | None = None
        if invalid_reason:
            last_action_error = invalid_reason
            state.invalid_action_count += 1
            state.actions_taken.append(f"invalid_action({invalid_reason})")
            reward_components["invalid_action"] = -0.05
        else:
            self._record_action(parsed_action)

            if isinstance(parsed_action, QueryLogsAction):
                self._handle_query_logs(parsed_action, reward_components)
            elif isinstance(parsed_action, QueryMetricsAction):
                self._handle_query_metrics(parsed_action, reward_components)
            elif isinstance(parsed_action, ReadRunbookAction):
                self._handle_read_runbook(parsed_action, reward_components)
            elif isinstance(parsed_action, InspectDependencyAction):
                self._handle_inspect_dependency(parsed_action, reward_components)
            elif isinstance(parsed_action, ClassifySeverityAction):
                self._handle_classify_severity(parsed_action, reward_components)
            elif isinstance(parsed_action, EscalateTeamAction):
                self._handle_escalate_team(parsed_action, reward_components)
            elif isinstance(parsed_action, ApplyMitigationAction):
                self._handle_apply_mitigation(parsed_action, reward_components)
            elif isinstance(parsed_action, PostStatusUpdateAction):
                self._handle_post_status_update(parsed_action, reward_components)
            elif isinstance(parsed_action, ResolveIncidentAction):
                self._handle_resolve_incident(parsed_action, reward_components)

        done, reason = self._compute_done()

        reward_value = round(sum(reward_components.values()), 6)
        info: dict[str, object] = {
            "reason": reason,
            "step_index": state.step_index,
            "max_steps": state.max_steps,
            "customer_impact_remaining": state.customer_impact_remaining,
            "harmful_action_count": state.harmful_action_count,
            "invalid_action_count": state.invalid_action_count,
            "last_action_error": last_action_error,
        }

        if done:
            info["final_score"] = grade_episode(state)

        return StepResult(
            observation=self._build_observation(),
            reward=StepReward(value=reward_value, components=reward_components),
            done=done,
            info=info,
        )

    # ---------------------------------------------------------------------
    # Action handlers
    # ---------------------------------------------------------------------

    def _handle_query_logs(
        self, action: QueryLogsAction, reward_components: dict[str, float]
    ) -> None:
        state = self._state
        assert state is not None

        matched_keys = []
        query_lower = action.query.lower()
        service_lower = action.service.lower()

        for key, snippets in state.evidence_pool_logs.items():
            if any(snippet.service.lower() == service_lower for snippet in snippets):
                if self._query_matches_log_key(query_lower, key, snippets):
                    matched_keys.append(key)

        if not matched_keys:
            self._penalize_irrelevant("irrelevant_investigation", reward_components)
            return

        revealed_any = False
        for key in matched_keys:
            snippets = state.evidence_pool_logs.get(key, [])
            newly_added = self._append_unique_logs(snippets)
            if newly_added > 0:
                revealed_any = True
                self._mark_evidence_revealed(key, reward_components)

        if not revealed_any:
            reward_components.setdefault("duplicate_or_low_value_query", -0.02)

    def _handle_query_metrics(
        self, action: QueryMetricsAction, reward_components: dict[str, float]
    ) -> None:
        state = self._state
        assert state is not None

        matched_keys = []
        requested_metric = action.metric_name.lower()
        requested_service = action.service.lower()

        for key, metric_list in state.evidence_pool_metrics.items():
            for metric in metric_list:
                if (
                    metric.service.lower() == requested_service
                    and requested_metric in metric.metric_name.lower()
                ):
                    matched_keys.append(key)

        if not matched_keys:
            self._penalize_irrelevant("irrelevant_investigation", reward_components)
            return

        revealed_any = False
        for key in matched_keys:
            metric_list = state.evidence_pool_metrics.get(key, [])
            newly_added = self._append_unique_metrics(metric_list)
            if newly_added > 0:
                revealed_any = True
                self._mark_evidence_revealed(key, reward_components)

        if not revealed_any:
            reward_components.setdefault("duplicate_or_low_value_query", -0.02)

    def _handle_read_runbook(
        self, action: ReadRunbookAction, reward_components: dict[str, float]
    ) -> None:
        state = self._state
        assert state is not None

        topic = action.topic.lower()
        matched_keys = []

        for key, notes in state.evidence_pool_runbooks.items():
            if topic in key.lower():
                matched_keys.append(key)
            elif any(topic in note.lower() for note in notes):
                matched_keys.append(key)

        if not matched_keys:
            if "general" in state.evidence_pool_runbooks:
                matched_keys = ["general"]

        if not matched_keys:
            self._penalize_irrelevant("irrelevant_investigation", reward_components)
            return

        revealed_any = False
        for key in matched_keys:
            notes = state.evidence_pool_runbooks.get(key, [])
            newly_added = self._append_unique_strings(notes, target="runbooks")
            if newly_added > 0:
                revealed_any = True
                runbook_key = self._map_runbook_key_to_evidence_key(key)
                if runbook_key:
                    self._mark_evidence_revealed(runbook_key, reward_components)

        if not revealed_any:
            reward_components.setdefault("duplicate_or_low_value_query", -0.02)

    def _handle_inspect_dependency(
        self, action: InspectDependencyAction, reward_components: dict[str, float]
    ) -> None:
        state = self._state
        assert state is not None

        chain = state.dependency_chain
        expected_pairs = {(chain[i], chain[i + 1]) for i in range(len(chain) - 1)}

        if (action.source_service, action.target_service) in expected_pairs:
            added_logs = self._append_unique_logs(state.evidence_pool_logs.get("dependency_log", []))
            if added_logs > 0:
                self._mark_evidence_revealed("dependency_log", reward_components)
            else:
                reward_components.setdefault("duplicate_or_low_value_query", -0.02)
            return

        self._penalize_irrelevant("irrelevant_dependency_inspection", reward_components)

    def _handle_classify_severity(
        self, action: ClassifySeverityAction, reward_components: dict[str, float]
    ) -> None:
        state = self._state
        assert state is not None

        state.classified_severity = action.severity

        if action.severity == state.true_severity:
            self._award_once("correct_severity", 0.10, reward_components)
            state.current_hypotheses = self._bounded_append(
                state.current_hypotheses,
                f"Severity classified as {action.severity}: {action.justification}",
            )
        else:
            reward_components["wrong_severity"] = -0.05
            state.invalid_action_count += 1

    def _handle_escalate_team(
        self, action: EscalateTeamAction, reward_components: dict[str, float]
    ) -> None:
        state = self._state
        assert state is not None

        state.escalated_team = action.team

        if not state.escalation_required:
            reward_components["unnecessary_escalation"] = -0.05
            return

        correct_team = state.expected_escalation_team or self._expected_team_for_state()
        if action.team == correct_team:
            state.escalation_done = True
            self._award_once("correct_escalation", 0.15, reward_components)
            state.current_hypotheses = self._bounded_append(
                state.current_hypotheses,
                f"Escalated to {action.team}: {action.justification}",
            )
        else:
            reward_components["wrong_escalation"] = -0.05
            state.invalid_action_count += 1

    def _handle_apply_mitigation(
        self, action: ApplyMitigationAction, reward_components: dict[str, float]
    ) -> None:
        state = self._state
        assert state is not None

        state.applied_mitigation = action.mitigation
        state.applied_mitigation_target = action.target_service

        valid_mitigation = action.mitigation in state.valid_mitigations
        target_is_relevant = action.target_service in state.dependency_chain or (
            action.target_service in state.impacted_services
        )

        if not target_is_relevant:
            self._apply_harmful_penalty("harmful_wrong_target", reward_components)
            return

        if valid_mitigation:
            state.mitigation_applied = True
            state.mitigation_effective = True

            impact_reduction = self._impact_reduction_for_mitigation(action.mitigation)
            state.customer_impact_remaining = max(
                0.0, round(state.customer_impact_remaining - impact_reduction, 3)
            )

            self._award_once("effective_mitigation", 0.20, reward_components)
            state.current_hypotheses = self._bounded_append(
                state.current_hypotheses,
                f"Applied mitigation {action.mitigation} on {action.target_service}",
            )

            if action.mitigation in {"scale_up", "shift_traffic", "clear_queue"}:
                self._append_unique_logs(state.evidence_pool_logs.get("root_cause_log", []))
                if "root_cause_log" in state.required_evidence_keys:
                    self._mark_evidence_revealed("root_cause_log", reward_components, silent=True)

        else:
            if action.mitigation in {"restart_service", "rollback_deploy", "toggle_feature_flag"}:
                if state.task_id in {"medium", "hard"}:
                    reward_components["partial_or_wrong_mitigation"] = -0.05
                    state.invalid_action_count += 1
                    state.customer_impact_remaining = min(
                        1.0, round(state.customer_impact_remaining - 0.05, 3)
                    )
                else:
                    if action.mitigation in state.valid_mitigations:
                        state.mitigation_applied = True
                        state.mitigation_effective = True
                        state.customer_impact_remaining = max(
                            0.0, round(state.customer_impact_remaining - 0.5, 3)
                        )
                        self._award_once("effective_mitigation", 0.20, reward_components)
                    else:
                        reward_components["wrong_mitigation"] = -0.05
                        state.invalid_action_count += 1
            else:
                self._apply_harmful_penalty("harmful_mitigation", reward_components)

    def _handle_post_status_update(
        self, action: PostStatusUpdateAction, reward_components: dict[str, float]
    ) -> None:
        state = self._state
        assert state is not None

        message = action.message.lower()

        useful = any(
            token in message
            for token in [
                "investigating",
                "mitigation",
                "impact",
                "root cause",
                "degraded",
                "resolved",
                "monitoring",
            ]
        )

        if useful and len(action.message.strip()) >= 20:
            self._award_once("useful_status_update", 0.03, reward_components)
        else:
            reward_components["low_value_status_update"] = -0.01

    def _handle_resolve_incident(
        self, action: ResolveIncidentAction, reward_components: dict[str, float]
    ) -> None:
        state = self._state
        assert state is not None

        state.resolution_attempted = True

        root_cause_correct = self._resolution_root_cause_matches(action.suspected_root_cause)
        impact_resolved = action.customer_impact_status == "resolved"
        evidence_ready = self._has_sufficient_resolution_evidence()

        if root_cause_correct and evidence_ready:
            if state.escalation_required and not state.escalation_done:
                reward_components["premature_resolve_missing_escalation"] = -0.10
                state.invalid_action_count += 1
                return

            if state.customer_impact_remaining > 0.25 and not state.mitigation_effective:
                reward_components["premature_resolve_unmitigated"] = -0.10
                state.invalid_action_count += 1
                return

            if not impact_resolved:
                reward_components["unclear_customer_impact_status"] = -0.05
                state.invalid_action_count += 1
                return

            state.resolved = True
            state.customer_impact_remaining = 0.0
            self._award_once("correct_resolution", 0.30, reward_components)
            state.current_hypotheses = self._bounded_append(
                state.current_hypotheses,
                f"Resolved: {action.resolution_summary}",
            )
            return

        reward_components["premature_or_wrong_resolution"] = -0.10
        state.invalid_action_count += 1

    # ---------------------------------------------------------------------
    # Reward and done helpers
    # ---------------------------------------------------------------------

    def _award_once(
        self,
        milestone: str,
        value: float,
        reward_components: dict[str, float],
    ) -> None:
        if milestone in self._milestones_awarded:
            return
        self._milestones_awarded.add(milestone)
        reward_components[milestone] = value

    def _mark_evidence_revealed(
        self,
        evidence_key: str,
        reward_components: dict[str, float],
        silent: bool = False,
    ) -> None:
        state = self._state
        assert state is not None

        if evidence_key not in state.revealed_evidence_keys:
            state.revealed_evidence_keys.append(evidence_key)
            if not silent:
                self._award_once(
                    f"evidence:{evidence_key}",
                    0.10,
                    reward_components,
                )

    def _penalize_irrelevant(
        self, key: str, reward_components: dict[str, float]
    ) -> None:
        reward_components[key] = reward_components.get(key, 0.0) - 0.05
        state = self._state
        assert state is not None
        state.invalid_action_count += 1

    def _apply_harmful_penalty(
        self, key: str, reward_components: dict[str, float]
    ) -> None:
        reward_components[key] = reward_components.get(key, 0.0) - 0.10
        state = self._state
        assert state is not None
        state.harmful_action_count += 1

    def _compute_done(self) -> tuple[bool, str]:
        state = self._state
        assert state is not None

        if state.resolved:
            return True, "resolved"

        if state.harmful_action_count >= 2:
            return True, "catastrophic_harmful_actions"

        if state.invalid_action_count >= 3:
            return True, "too_many_invalid_actions"

        if state.step_index >= state.max_steps:
            return True, "step_budget_exhausted"

        return False, "in_progress"

    # ---------------------------------------------------------------------
    # Observation builders
    # ---------------------------------------------------------------------

    def _build_observation(self) -> IncidentObservation:
        state = self._state
        assert state is not None

        return IncidentObservation(
            incident_id=state.incident_id,
            task_id=state.task_id,
            step_index=state.step_index,
            max_steps=state.max_steps,
            title=state.title,
            alert_summary=state.alert_summary,
            customer_impact_summary=self._current_customer_impact_summary(),
            visible_metrics=self._bounded_metrics(state.visible_metrics),
            visible_logs=self._bounded_logs(state.visible_logs),
            visible_runbook_notes=state.visible_runbook_notes[-3:],
            actions_taken=state.actions_taken[-6:],
            current_hypotheses=state.current_hypotheses[-4:],
            available_actions=self.AVAILABLE_ACTIONS,
        )

    def _current_customer_impact_summary(self) -> str:
        state = self._state
        assert state is not None

        if state.customer_impact_remaining <= 0.0:
            return "Customer impact has been resolved."
        if state.customer_impact_remaining <= 0.3:
            return "Customer impact is low and appears mitigated, but continued verification is needed."
        if state.customer_impact_remaining <= 0.6:
            return (
                "Customer impact remains moderate. Some symptoms may be reduced but the incident is not fully resolved."
            )
        return state.customer_impact_summary

    # ---------------------------------------------------------------------
    # Append / dedupe helpers
    # ---------------------------------------------------------------------

    def _append_unique_logs(self, snippets: Iterable[LogSnippet]) -> int:
        state = self._state
        assert state is not None

        existing = {(log.service, log.timestamp, log.message) for log in state.visible_logs}
        added = 0
        for log in snippets:
            key = (log.service, log.timestamp, log.message)
            if key not in existing:
                state.visible_logs.append(log)
                existing.add(key)
                added += 1
        state.visible_logs = self._bounded_logs(state.visible_logs)
        return added

    def _append_unique_metrics(self, metrics: Iterable[MetricSnapshot]) -> int:
        state = self._state
        assert state is not None

        existing = {(m.service, m.metric_name) for m in state.visible_metrics}
        added = 0
        for metric in metrics:
            key = (metric.service, metric.metric_name)
            if key not in existing:
                state.visible_metrics.append(metric)
                existing.add(key)
                added += 1
        state.visible_metrics = self._bounded_metrics(state.visible_metrics)
        return added

    def _append_unique_strings(self, values: Iterable[str], target: str) -> int:
        state = self._state
        assert state is not None

        if target != "runbooks":
            raise ValueError(f"Unsupported string target: {target}")

        existing = set(state.visible_runbook_notes)
        added = 0
        for value in values:
            if value not in existing:
                state.visible_runbook_notes.append(value)
                existing.add(value)
                added += 1
        state.visible_runbook_notes = state.visible_runbook_notes[-3:]
        return added

    def _bounded_logs(self, logs: list[LogSnippet]) -> list[LogSnippet]:
        return logs[-6:]

    def _bounded_metrics(self, metrics: list[MetricSnapshot]) -> list[MetricSnapshot]:
        return metrics[-5:]

    def _bounded_append(self, values: list[str], value: str, limit: int = 4) -> list[str]:
        new_values = list(values)
        if value not in new_values:
            new_values.append(value)
        return new_values[-limit:]

    # ---------------------------------------------------------------------
    # Semantic helpers
    # ---------------------------------------------------------------------

    def _expected_team_for_state(self) -> str:
        state = self._state
        assert state is not None

        root = state.root_cause.lower()
        if "db" in root or "postgres" in root or "database" in root:
            return "database"
        if "deploy" in root or "feature flag" in root:
            return "backend"
        if "inventory" in root:
            return "sre"
        return "backend"

    def _impact_reduction_for_mitigation(self, mitigation: str) -> float:
        state = self._state
        assert state is not None

        if state.task_id == "easy":
            if mitigation in {"rollback_deploy", "toggle_feature_flag"}:
                return 1.0
            return 0.4

        if state.task_id == "medium":
            if mitigation in {"scale_up", "shift_traffic"}:
                return 0.75
            return 0.2

        if mitigation == "clear_queue":
            return 0.35
        if mitigation in {"scale_up", "shift_traffic"}:
            return 0.55
        return 0.2

    def _resolution_root_cause_matches(self, proposed: str) -> bool:
        state = self._state
        assert state is not None

        proposed_lower = proposed.lower()
        root_lower = state.root_cause.lower()

        key_terms = [token for token in root_lower.replace("-", " ").split() if len(token) > 4]
        overlap = sum(1 for token in key_terms if token in proposed_lower)

        if root_lower in proposed_lower:
            return True
        return overlap >= max(2, min(4, len(key_terms) // 2))

    def _has_sufficient_resolution_evidence(self) -> bool:
        state = self._state
        assert state is not None

        required = set(state.required_evidence_keys)
        revealed = set(state.revealed_evidence_keys)

        if not required:
            return True

        ratio = len(required.intersection(revealed)) / len(required)
        return ratio >= 0.6

    def _query_matches_log_key(self, query_lower: str, key: str, snippets: list) -> bool:
        if key.replace("_", " ") in query_lower:
            return True

        for token in [
            "error",
            "latency",
            "timeout",
            "deploy",
            "rollback",
            "db",
            "database",
            "dependency",
            "queue",
            "cpu",
        ]:
            if token in query_lower:
                if token in key.lower():
                    return True
                if any(token in snippet.message.lower() for snippet in snippets):
                    return True

        if query_lower in {"recent", "latest", "incident", "errors", "logs"}:
            return True

        return False

    def _map_runbook_key_to_evidence_key(self, runbook_key: str) -> str | None:
        mapping = {
            "deploys": "rollback_hint",
            "database": "db_runbook_hint",
            "dependencies": "dependency_log",
            "causal": "causal_runbook_hint",
        }
        return mapping.get(runbook_key)

    def _record_action(self, action: IncidentAction) -> None:
        state = self._state
        assert state is not None

        if isinstance(action, QueryLogsAction):
            text = f"query_logs(service={action.service}, query={action.query}, window_min={action.window_min})"
        elif isinstance(action, QueryMetricsAction):
            text = f"query_metrics(service={action.service}, metric_name={action.metric_name}, window_min={action.window_min})"
        elif isinstance(action, ReadRunbookAction):
            text = f"read_runbook(service={action.service}, topic={action.topic})"
        elif isinstance(action, InspectDependencyAction):
            text = (
                f"inspect_dependency(source_service={action.source_service}, "
                f"target_service={action.target_service})"
            )
        elif isinstance(action, ClassifySeverityAction):
            text = f"classify_severity(severity={action.severity})"
        elif isinstance(action, EscalateTeamAction):
            text = f"escalate_team(team={action.team}, urgency={action.urgency})"
        elif isinstance(action, ApplyMitigationAction):
            text = f"apply_mitigation(mitigation={action.mitigation}, target_service={action.target_service})"
        elif isinstance(action, PostStatusUpdateAction):
            text = f"post_status_update(audience={action.audience})"
        elif isinstance(action, ResolveIncidentAction):
            text = "resolve_incident(...)"
        else:
            text = action.__class__.__name__

        state.actions_taken.append(text)
        state.actions_taken = state.actions_taken[-6:]

    # ---------------------------------------------------------------------
    # Parsing / validation
    # ---------------------------------------------------------------------

    def _parse_action(self, action: IncidentAction | dict) -> tuple[IncidentAction | None, str | None]:
        if isinstance(
            action,
            (
                QueryLogsAction,
                QueryMetricsAction,
                ReadRunbookAction,
                InspectDependencyAction,
                ClassifySeverityAction,
                EscalateTeamAction,
                ApplyMitigationAction,
                PostStatusUpdateAction,
                ResolveIncidentAction,
            ),
        ):
            return action, None

        if isinstance(action, dict):
            try:
                return self._validate_action_dict(action), None
            except ValidationError as exc:
                return None, exc.errors()[0].get("msg", "invalid action")

        return None, f"unsupported_action_type:{type(action)}"

    def _validate_action_dict(self, payload: dict) -> IncidentAction:
        action_type = payload.get("action_type")

        mapping = {
            "query_logs": QueryLogsAction,
            "query_metrics": QueryMetricsAction,
            "read_runbook": ReadRunbookAction,
            "inspect_dependency": InspectDependencyAction,
            "classify_severity": ClassifySeverityAction,
            "escalate_team": EscalateTeamAction,
            "apply_mitigation": ApplyMitigationAction,
            "post_status_update": PostStatusUpdateAction,
            "resolve_incident": ResolveIncidentAction,
        }

        model_cls = mapping.get(action_type)
        if model_cls is None:
            raise ValidationError.from_exception_data(
                title="IncidentAction",
                line_errors=[],
            )
        return model_cls.model_validate(payload)

    # ---------------------------------------------------------------------
    # Task loading
    # ---------------------------------------------------------------------

    def _load_task_config(self, task_id: str) -> TaskConfig:
        tasks_dir = Path(__file__).resolve().parent / "tasks"
        path = tasks_dir / f"{task_id}_incident.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing task config: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return TaskConfig.model_validate(raw)

    def _build_state_from_config(self, config: TaskConfig) -> IncidentState:
        evidence_pool_logs: dict[str, list[LogSnippet]] = {}
        for entry in config.evidence.logs:
            evidence_pool_logs.setdefault(entry.evidence_key, []).append(entry.log)

        evidence_pool_metrics: dict[str, list[MetricSnapshot]] = {}
        for entry in config.evidence.metrics:
            evidence_pool_metrics.setdefault(entry.evidence_key, []).append(entry.metric)

        evidence_pool_runbooks: dict[str, list[str]] = {}
        for entry in config.evidence.runbooks:
            evidence_pool_runbooks.setdefault(entry.evidence_key, []).extend(entry.notes)

        incident_id = f"{config.task_id}-{config.seed}"

        state = IncidentState(
            incident_id=incident_id,
            task_id=config.task_id,
            title=config.title,
            true_severity=config.true_severity,
            root_cause=config.root_cause,
            impacted_services=config.impacted_services,
            dependency_chain=config.dependency_chain,
            misleading_signals=config.misleading_signals,
            required_evidence_keys=config.required_evidence_keys,
            revealed_evidence_keys=[],
            valid_mitigations=config.valid_mitigations,
            escalation_required=config.escalation_required,
            expected_escalation_team=config.expected_escalation_team,
            escalation_done=False,
            mitigation_applied=False,
            mitigation_effective=False,
            customer_impact_remaining=1.0,
            harmful_action_count=0,
            invalid_action_count=0,
            step_index=0,
            max_steps=config.max_steps,
            resolved=False,
            evidence_pool_logs=evidence_pool_logs,
            evidence_pool_metrics=evidence_pool_metrics,
            evidence_pool_runbooks=evidence_pool_runbooks,
            visible_logs=[],
            visible_metrics=[],
            visible_runbook_notes=[],
            actions_taken=[],
            current_hypotheses=[],
            customer_impact_summary=config.customer_impact_summary,
            alert_summary=config.alert_summary,
        )

        self._apply_initial_visibility(state, config)
        return state

    def _apply_initial_visibility(self, state: IncidentState, config: TaskConfig) -> None:
        for key in config.initial_visible.log_keys:
            for log in state.evidence_pool_logs.get(key, []):
                state.visible_logs.append(log)
            if key not in state.revealed_evidence_keys:
                state.revealed_evidence_keys.append(key)

        for key in config.initial_visible.metric_keys:
            for metric in state.evidence_pool_metrics.get(key, []):
                state.visible_metrics.append(metric)
            if key not in state.revealed_evidence_keys:
                state.revealed_evidence_keys.append(key)

        for key in config.initial_visible.runbook_keys:
            for note in state.evidence_pool_runbooks.get(key, []):
                state.visible_runbook_notes.append(note)
            mapped_key = self._map_runbook_key_to_evidence_key(key)
            if mapped_key and mapped_key not in state.revealed_evidence_keys:
                state.revealed_evidence_keys.append(mapped_key)
            elif key not in state.revealed_evidence_keys:
                state.revealed_evidence_keys.append(key)

        state.visible_logs = self._bounded_logs(state.visible_logs)
        state.visible_metrics = self._bounded_metrics(state.visible_metrics)
        state.visible_runbook_notes = state.visible_runbook_notes[-3:]

    def _require_state(self) -> None:
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
