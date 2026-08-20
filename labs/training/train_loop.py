"""Inspectable train/eval/checkpoint loop for the R4 tiny LM.

The loop intentionally keeps the optimizer boundary visible. Gradient
accumulation is normalized by the total number of valid shifted targets, so
micro-batches with different padding/mask counts do not silently change the
objective.

References:
- https://docs.pytorch.org/docs/stable/autograd.html
- https://docs.pytorch.org/docs/stable/optim.html
- https://docs.pytorch.org/docs/stable/notes/serialization.html
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from labs.training.label_shift import causal_lm_loss_sum_and_count


Batch = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class StepStats:
    """Observable values for one optimizer update."""

    loss: float
    valid_tokens: int
    grad_norm: float


@dataclass(frozen=True)
class EpochStats:
    """Observable values for one pass over the training batches."""

    epoch: int
    train_loss: float
    eval_loss: float | None
    optimizer_steps: int


def _validate_batch(batch: Batch) -> None:
    input_ids, labels = batch
    if input_ids.ndim != 2 or labels.ndim != 2:
        raise ValueError("input_ids and labels must both have shape [B, T]")
    if input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have the same shape")
    if input_ids.shape[0] == 0 or input_ids.shape[1] < 2:
        raise ValueError("batches need at least one sample and two tokens")


def _model_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _divide_gradients(parameters: Sequence[nn.Parameter], denominator: int) -> None:
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(denominator)


def train_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    batches: Sequence[Batch],
    *,
    grad_accum_steps: int = 1,
    max_grad_norm: float | None = 1.0,
) -> list[StepStats]:
    """Run one training epoch and return one record per optimizer step.

    Each batch contributes a *summed* token loss to the gradient. The gradient
    is divided once by the total valid-token count before clipping and stepping.
    This makes accumulation match a single concatenated batch under the same
    model and optimizer state.
    """

    if not batches:
        raise ValueError("at least one training batch is required")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive or None")

    parameters = _model_parameters(model)
    if not parameters:
        raise ValueError("model must expose at least one trainable parameter")
    model.train()
    records: list[StepStats] = []

    for start in range(0, len(batches), grad_accum_steps):
        group = batches[start : start + grad_accum_steps]
        optimizer.zero_grad(set_to_none=True)
        total_loss_sum: Tensor | None = None
        total_valid_tokens = 0

        for batch in group:
            _validate_batch(batch)
            input_ids, labels = batch
            logits = model(input_ids)
            loss_sum, valid_tokens = causal_lm_loss_sum_and_count(logits, labels)
            # Backward on the sum first; normalize exactly after all micro-batches.
            loss_sum.backward()
            total_loss_sum = loss_sum.detach() if total_loss_sum is None else total_loss_sum + loss_sum.detach()
            total_valid_tokens += valid_tokens

        if total_loss_sum is None or total_valid_tokens == 0:
            raise ValueError("the accumulation group produced no valid targets")
        _divide_gradients(parameters, total_valid_tokens)
        if max_grad_norm is None:
            squared_norm = sum(
                parameter.grad.float().pow(2).sum()
                for parameter in parameters
                if parameter.grad is not None
            )
            grad_norm = squared_norm.sqrt()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        optimizer.step()
        records.append(
            StepStats(
                loss=float((total_loss_sum / total_valid_tokens).cpu()),
                valid_tokens=total_valid_tokens,
                grad_norm=float(grad_norm.cpu()),
            )
        )
    return records


@torch.no_grad()
def evaluate(model: nn.Module, batches: Sequence[Batch]) -> float:
    """Return valid-token mean causal loss without updating model state."""

    if not batches:
        raise ValueError("at least one evaluation batch is required")
    model.eval()
    total_loss = 0.0
    total_valid_tokens = 0
    for batch in batches:
        _validate_batch(batch)
        input_ids, labels = batch
        loss_sum, valid_tokens = causal_lm_loss_sum_and_count(model(input_ids), labels)
        total_loss += float(loss_sum.cpu())
        total_valid_tokens += valid_tokens
    if total_valid_tokens == 0:
        raise ValueError("evaluation batches produced no valid targets")
    return total_loss / total_valid_tokens


def fit(
    model: nn.Module,
    optimizer: Optimizer,
    train_batches: Sequence[Batch],
    *,
    epochs: int,
    eval_batches: Sequence[Batch] | None = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float | None = 1.0,
    scheduler: object | None = None,
) -> list[EpochStats]:
    """Train for a fixed number of epochs and return compact history."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    history: list[EpochStats] = []
    for epoch in range(1, epochs + 1):
        step_records = train_epoch(
            model,
            optimizer,
            train_batches,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
        )
        if scheduler is not None:
            step = getattr(scheduler, "step", None)
            if step is None or not callable(step):
                raise TypeError("scheduler must expose a callable step()")
            step()
        eval_loss = evaluate(model, eval_batches) if eval_batches is not None else None
        total_tokens = sum(record.valid_tokens for record in step_records)
        weighted_loss = sum(record.loss * record.valid_tokens for record in step_records)
        history.append(
            EpochStats(
                epoch=epoch,
                train_loss=weighted_loss / total_tokens,
                eval_loss=eval_loss,
                optimizer_steps=len(step_records),
            )
        )
    return history


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    step: int,
    config: Mapping[str, object] | None = None,
) -> None:
    """Save model/optimizer/RNG state needed for a resumable toy run."""

    if step < 0:
        raise ValueError("step must be non-negative")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": dict(config or {}),
        "rng_state": torch.get_rng_state(),
    }
    torch.save(payload, target)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, object]:
    """Load a checkpoint and optionally restore the CPU RNG state."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or "model" not in payload or "optimizer" not in payload:
        raise ValueError("checkpoint must contain model and optimizer state")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if restore_rng and "rng_state" in payload:
        rng_state = payload["rng_state"]
        if not isinstance(rng_state, Tensor):
            raise ValueError("checkpoint rng_state must be a tensor")
        torch.set_rng_state(rng_state)
    return payload
