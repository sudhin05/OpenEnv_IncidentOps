from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator


class MetricSnapshot(BaseModel):
    service: str
    metric_name: str
    value: float
    unit: str
    trend: Literal["up", "down", "flat"]
    summary: str


class LogSnippet(BaseModel):
    service: str
    timestamp: str
    severity: Literal["info", "warning", "error"]
    message: str
    relevance_hint: str | None = None


class IncidentObservation(BaseModel):
    incident_id: str
    task_id: str
    step_index: int
    max_steps: int
    title: str
    alert_summary: str
    customer_impact_summary: str
    visible_metrics: list[MetricSnapshot]
    visible_logs: list[LogSnippet]
    visible_runbook_notes: list[str]
    actions_taken: list[str]
    current_hypotheses: list[str]
    available_actions: list[str]


class QueryLogsAction(BaseModel):
    action_type: Literal["query_logs"]
    service: str
    query: str
    window_min: int = Field(ge=1, le=240)


class QueryMetricsAction(BaseModel):
    action_type: Literal["query_metrics"]
    service: str
    metric_name: str
    window_min: int = Field(ge=1, le=240)


class ReadRunbookAction(BaseModel):
    action_type: Literal["read_runbook"]
    service: str
    topic: str


class InspectDependencyAction(BaseModel):
    action_type: Literal["inspect_dependency"]
    source_service: str
    target_service: str


class ClassifySeverityAction(BaseModel):
    action_type: Literal["classify_severity"]
    severity: Literal["sev1", "sev2", "sev3"]
    justification: str


class EscalateTeamAction(BaseModel):
    action_type: Literal["escalate_team"]
    team: Literal["backend", "database", "infra", "payments", "frontend", "sre"]
    urgency: Literal["low", "medium", "high"]
    justification: str


class ApplyMitigationAction(BaseModel):
    action_type: Literal["apply_mitigation"]
    mitigation: Literal[
        "restart_service",
        "rollback_deploy",
        "shift_traffic",
        "clear_queue",
        "scale_up",
        "toggle_feature_flag",
    ]
    target_service: str
    justification: str


class PostStatusUpdateAction(BaseModel):
    action_type: Literal["post_status_update"]
    audience: Literal["internal", "customer"]
    message: str


class ResolveIncidentAction(BaseModel):
    action_type: Literal["resolve_incident"]
    suspected_root_cause: str
    mitigation_applied: str
    customer_impact_status: Literal["resolved", "mitigated", "unknown"]
    resolution_summary: str
    confidence: float = Field(ge=0.0, le=1.0)


IncidentAction = Annotated[
    Union[
        QueryLogsAction,
        QueryMetricsAction,
        ReadRunbookAction,
        InspectDependencyAction,
        ClassifySeverityAction,
        EscalateTeamAction,
        ApplyMitigationAction,
        PostStatusUpdateAction,
        ResolveIncidentAction,
    ],
    Field(discriminator="action_type"),
]


class StepReward(BaseModel):
    value: float
    components: dict[str, float]


class IncidentState(BaseModel):
    incident_id: str
    task_id: str
    title: str
    true_severity: Literal["sev1", "sev2", "sev3"]
    root_cause: str
    impacted_services: list[str]
    dependency_chain: list[str]
    misleading_signals: list[str]
    required_evidence_keys: list[str]
    revealed_evidence_keys: list[str]
    valid_mitigations: list[str]
    escalation_required: bool
    expected_escalation_team: str | None = None
    escalation_done: bool = False
    mitigation_applied: bool = False
    mitigation_effective: bool = False
    classified_severity: Literal["sev1", "sev2", "sev3"] | None = None
    escalated_team: str | None = None
    applied_mitigation: str | None = None
    applied_mitigation_target: str | None = None
    resolution_attempted: bool = False
    customer_impact_remaining: float = Field(ge=0.0, le=1.0)
    harmful_action_count: int = Field(default=0, ge=0)
    invalid_action_count: int = Field(default=0, ge=0)
    step_index: int = Field(default=0, ge=0)
    max_steps: int = Field(ge=1)
    resolved: bool = False
    evidence_pool_logs: dict[str, list[LogSnippet]] = Field(default_factory=dict)
    evidence_pool_metrics: dict[str, list[MetricSnapshot]] = Field(default_factory=dict)
    evidence_pool_runbooks: dict[str, list[str]] = Field(default_factory=dict)
    visible_logs: list[LogSnippet] = Field(default_factory=list)
    visible_metrics: list[MetricSnapshot] = Field(default_factory=list)
    visible_runbook_notes: list[str] = Field(default_factory=list)
    actions_taken: list[str] = Field(default_factory=list)
    current_hypotheses: list[str] = Field(default_factory=list)
    customer_impact_summary: str = ""
    alert_summary: str = ""

    @field_validator("dependency_chain")
    @classmethod
    def dependency_chain_not_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("dependency_chain must not be empty")
        return value


class StepResult(BaseModel):
    observation: IncidentObservation
    reward: StepReward
    done: bool
    info: dict


class TaskEvidenceLog(BaseModel):
    evidence_key: str
    log: LogSnippet


class TaskEvidenceMetric(BaseModel):
    evidence_key: str
    metric: MetricSnapshot


class TaskEvidenceRunbook(BaseModel):
    evidence_key: str
    notes: list[str]


class TaskEvidenceBundle(BaseModel):
    logs: list[TaskEvidenceLog]
    metrics: list[TaskEvidenceMetric]
    runbooks: list[TaskEvidenceRunbook]


class TaskInitialVisible(BaseModel):
    log_keys: list[str] = Field(default_factory=list)
    metric_keys: list[str] = Field(default_factory=list)
    runbook_keys: list[str] = Field(default_factory=list)


class TaskConfig(BaseModel):
    task_id: Literal["easy", "medium", "hard"]
    seed: int
    max_steps: int
    title: str
    alert_summary: str
    customer_impact_summary: str
    true_severity: Literal["sev1", "sev2", "sev3"]
    root_cause: str
    impacted_services: list[str]
    dependency_chain: list[str]
    misleading_signals: list[str]
    valid_mitigations: list[str]
    escalation_required: bool
    expected_escalation_team: str | None = None
    required_evidence_keys: list[str]
    evidence: TaskEvidenceBundle
    initial_visible: TaskInitialVisible

