from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .env import IncidentOpsEnv
from .models import IncidentAction, IncidentObservation, IncidentState, StepResult

app = FastAPI(title="IncidentOpsEnv")

_env = IncidentOpsEnv(task_id="easy")


class ResetRequest(BaseModel):
    task_id: str | None = None


class StepRequest(BaseModel):
    action: IncidentAction | dict


@app.post("/reset", response_model=IncidentObservation)
def reset_env(payload: ResetRequest = ResetRequest()) -> IncidentObservation:
    return _env.reset(task_id=payload.task_id)


@app.post("/step", response_model=StepResult)
def step_env(payload: StepRequest) -> StepResult:
    return _env.step(payload.action)


import os
import uvicorn

@app.get("/state", response_model=IncidentState)
def get_state() -> IncidentState:
    return _env.state()


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()

