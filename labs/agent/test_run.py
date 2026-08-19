"""Tests for the dependency-free durable Agent run model."""

from __future__ import annotations

import unittest

from labs.agent.run import AgentState, DurableAgentRun, InvalidTransition, StepLimitExceeded


class DurableAgentRunTest(unittest.TestCase):
    def test_checkpoint_resume_and_valid_transitions(self) -> None:
        run = DurableAgentRun("run-1", max_steps=5)
        run.transition(AgentState.TOOL_PENDING, "selected search")
        run.transition(AgentState.TOOL_RUNNING)
        run.record_effect("search:1", {"items": ["a"]})
        restored = DurableAgentRun.restore(run.snapshot())
        self.assertEqual(restored.state, AgentState.TOOL_RUNNING)
        self.assertEqual(restored.record_effect("search:1", {"items": ["different"]}), {"items": ["a"]})
        restored.transition(AgentState.COMPLETED)
        self.assertEqual(restored.state, AgentState.COMPLETED)

    def test_invalid_transition_and_step_limit(self) -> None:
        run = DurableAgentRun("run-2", max_steps=1)
        with self.assertRaises(InvalidTransition):
            run.transition(AgentState.TOOL_RUNNING)
        run.transition(AgentState.COMPLETED)
        with self.assertRaises(StepLimitExceeded):
            run.transition(AgentState.FAILED)

    def test_waiting_retry_can_resume_to_tool_pending(self) -> None:
        run = DurableAgentRun("run-3")
        run.transition(AgentState.TOOL_PENDING)
        run.transition(AgentState.TOOL_RUNNING)
        run.transition(AgentState.WAITING_RETRY, "timeout")
        run.transition(AgentState.TOOL_PENDING, "backoff elapsed")
        self.assertEqual(run.state, AgentState.TOOL_PENDING)


if __name__ == "__main__":
    unittest.main()

