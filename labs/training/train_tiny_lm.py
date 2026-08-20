"""Run a tiny deterministic decoder-LM experiment from the command line.

Examples:

    .venv/bin/python -m labs.training.train_tiny_lm --epochs 5
    .venv/bin/python -m labs.training.train_tiny_lm --device mps

The default is CPU so the result is easy to reproduce on the learner's Mac.
MPS is an optional smoke-test lane, not a CUDA performance benchmark.
"""

from __future__ import annotations

import argparse

import torch

from labs.training.tiny_lm import TinyDecoderLM
from labs.training.train_loop import Batch, evaluate, fit


def build_repeating_batches(
    *,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    num_batches: int,
    device: torch.device,
) -> list[Batch]:
    """Build a tiny periodic corpus where each token predicts the next cycle."""

    rows = []
    for row_id in range(batch_size * num_batches):
        rows.append([(row_id + offset) % vocab_size for offset in range(seq_len)])
    tokens = torch.tensor(rows, dtype=torch.long, device=device)
    return [
        (tokens[start : start + batch_size], tokens[start : start + batch_size].clone())
        for start in range(0, len(tokens), batch_size)
    ]


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS requested, but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    raise ValueError(f"unsupported device: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model = TinyDecoderLM(
        vocab_size=8,
        max_seq_len=8,
        d_model=16,
        num_heads=4,
        num_kv_heads=2,
        ffn_dim=32,
    ).to(device)
    batches = build_repeating_batches(
        vocab_size=8,
        seq_len=8,
        batch_size=4,
        num_batches=4,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    before = evaluate(model, batches)
    history = fit(
        model,
        optimizer,
        batches,
        epochs=args.epochs,
        eval_batches=batches,
        grad_accum_steps=args.grad_accum_steps,
    )

    print(f"device={device} initial_loss={before:.4f}")
    for item in history:
        print(
            f"epoch={item.epoch} train_loss={item.train_loss:.4f} "
            f"eval_loss={item.eval_loss:.4f} optimizer_steps={item.optimizer_steps}"
        )


if __name__ == "__main__":
    main()
