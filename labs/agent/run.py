"""A dependency-free durable Agent run model for interview practice."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AgentState(StrEnum):
    PLANNING = "PLANNING"
    TOOL_PENDING = "TOOL_PENDING"
    TOOL_RUNNING = "TOOL_RUNNING"
    WAITING_RETRY = "WAITING_RETRY"
    WAITING_HUMAN = "WAITING_HUMAN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class InvalidTransition(ValueError):
    pass


class StepLimitExceeded(RuntimeError):
    pass


_ALLOWED_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.PLANNING: {AgentState.TOOL_PENDING, AgentState.COMPLETED, AgentState.FAILED},
    AgentState.TOOL_PENDING: {AgentState.TOOL_RUNNING, AgentState.FAILED},
    AgentState.TOOL_RUNNING: {
        AgentState.PLANNING,
        AgentState.WAITING_RETRY,
        AgentState.WAITING_HUMAN,
        AgentState.COMPLETED,
        AgentState.FAILED,
    },
    AgentState.WAITING_RETRY: {AgentState.TOOL_PENDING, AgentState.FAILED},
    AgentState.WAITING_HUMAN: {AgentState.TOOL_PENDING, AgentState.COMPLETED, AgentState.FAILED},
    AgentState.COMPLETED: set(),
    AgentState.FAILED: set(),
}


@dataclass
class DurableAgentRun:
    run_id: str
    max_steps: int = 20
    state: AgentState = AgentState.PLANNING
    step: int = 0
    history: list[str] = field(default_factory=list)
    effects: dict[str, Any] = field(default_factory=dict)

    def transition(self, next_state: AgentState, reason: str = "") -> None:
        if self.step >= self.max_steps:
            raise StepLimitExceeded(f"run {self.run_id} exceeded max_steps={self.max_steps}")
        if next_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise InvalidTransition(f"{self.state} -> {next_state} is not allowed")
        self.step += 1
        self.state = next_state
        self.history.append(f"{self.step}:{next_state}:{reason}".rstrip(":"))

    def record_effect(self, effect_id: str, result: Any) -> Any:
        """Record an external effect once and return the durable result on retry."""

        if not effect_id:
            raise ValueError("effect_id must be non-empty")
        if effect_id not in self.effects:
            self.effects[effect_id] = result
        return self.effects[effect_id]

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "max_steps": self.max_steps,
            "state": self.state.value,
            "step": self.step,
            "history": list(self.history),
            "effects": dict(self.effects),
        }

    @classmethod
    def restore(cls, snapshot: dict[str, Any]) -> "DurableAgentRun":
        required = {"run_id", "max_steps", "state", "step", "history", "effects"}
        if not required.issubset(snapshot):
            raise ValueError("snapshot is missing required fields")
        run = cls(
            run_id=str(snapshot["run_id"]),
            max_steps=int(snapshot["max_steps"]),
            state=AgentState(snapshot["state"]),
            step=int(snapshot["step"]),
            history=list(snapshot["history"]),
            effects=dict(snapshot["effects"]),
        )
        if run.step < 0 or run.step > run.max_steps:
            raise ValueError("snapshot step is outside the allowed range")
        return run

