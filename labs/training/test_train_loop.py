"""Tests for R4 train/eval/checkpoint behavior."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from labs.training.tiny_lm import TinyDecoderLM
from labs.training.train_loop import (
    evaluate,
    fit,
    load_checkpoint,
    save_checkpoint,
    train_epoch,
)


def _batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    input_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0],
            [2, 3, 4, 0, 1],
            [3, 4, 0, 1, 2],
        ],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    return [(input_ids[:2], labels[:2]), (input_ids[2:], labels[2:])]


def _model() -> TinyDecoderLM:
    return TinyDecoderLM(
        vocab_size=5,
        max_seq_len=5,
        d_model=8,
        num_heads=2,
        ffn_dim=16,
    )


class TrainLoopTest(unittest.TestCase):
    def test_gradient_accumulation_matches_concatenated_batch(self) -> None:
        torch.manual_seed(0)
        accumulated = _model()
        concatenated = copy.deepcopy(accumulated)
        accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.05)
        concatenated_optimizer = torch.optim.SGD(concatenated.parameters(), lr=0.05)
        batches = _batches()

        accumulated_stats = train_epoch(
            accumulated,
            accumulated_optimizer,
            batches,
            grad_accum_steps=2,
            max_grad_norm=None,
        )
        concatenated_stats = train_epoch(
            concatenated,
            concatenated_optimizer,
            [(torch.cat([batches[0][0], batches[1][0]]), torch.cat([batches[0][1], batches[1][1]]))],
            max_grad_norm=None,
        )

        self.assertEqual(accumulated_stats[0].valid_tokens, 16)
        self.assertEqual(concatenated_stats[0].valid_tokens, 16)
        for actual, expected in zip(accumulated.parameters(), concatenated.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_fit_reports_finite_losses_and_evaluation(self) -> None:
        torch.manual_seed(1)
        model = _model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
        history = fit(
            model,
            optimizer,
            _batches(),
            epochs=3,
            eval_batches=_batches(),
            grad_accum_steps=2,
        )

        self.assertEqual(len(history), 3)
        self.assertTrue(all(torch.isfinite(torch.tensor(item.train_loss)) for item in history))
        self.assertTrue(all(item.eval_loss is not None for item in history))
        self.assertLess(history[-1].train_loss, history[0].train_loss)
        self.assertGreater(evaluate(model, _batches()), 0.0)

    def test_checkpoint_resume_matches_continuous_training(self) -> None:
        torch.manual_seed(2)
        initial = _model()
        continuous = copy.deepcopy(initial)
        resumed = copy.deepcopy(initial)
        continuous_optimizer = torch.optim.AdamW(continuous.parameters(), lr=0.02)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.02)
        batches = _batches()

        train_epoch(continuous, continuous_optimizer, batches[:1])
        train_epoch(resumed, resumed_optimizer, batches[:1])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "step-1.pt"
            save_checkpoint(checkpoint, resumed, resumed_optimizer, step=1, config={"stage": "R4"})
            restored = _model()
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.02)
            payload = load_checkpoint(checkpoint, restored, restored_optimizer)
            self.assertEqual(payload["step"], 1)
            self.assertEqual(payload["config"], {"stage": "R4"})
            train_epoch(continuous, continuous_optimizer, batches[1:])
            train_epoch(restored, restored_optimizer, batches[1:])

        for actual, expected in zip(restored.parameters(), continuous.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_rejects_empty_batches_and_invalid_accumulation(self) -> None:
        model = _model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with self.assertRaises(ValueError):
            train_epoch(model, optimizer, [])
        with self.assertRaises(ValueError):
            train_epoch(model, optimizer, _batches(), grad_accum_steps=0)
        with self.assertRaises(ValueError):
            fit(model, optimizer, _batches(), epochs=0)


if __name__ == "__main__":
    unittest.main()
