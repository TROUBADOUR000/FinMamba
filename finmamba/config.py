from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the refactored FinMamba model."
    )

    data = parser.add_argument_group("data")
    data.add_argument("--stock", type=str, default="nasdaq")
    data.add_argument("--data-dir", type=Path, default=Path("data"))
    data.add_argument(
        "--relation-dir",
        type=Path,
        default=Path("nasdaq_stock_relation"),
        help="Directory containing day{index}.pkl relation matrices.",
    )
    data.add_argument("--relation-pattern", type=str, default="day{index}.pkl")
    data.add_argument("--train-start", type=str, default="2018-01-01")
    data.add_argument("--train-end", type=str, default="2021-12-31")
    data.add_argument("--valid-start", type=str, default="2022-01-01")
    data.add_argument("--valid-end", type=str, default="2022-12-31")
    data.add_argument("--test-start", type=str, default="2023-01-01")
    data.add_argument("--test-end", type=str, default="2023-12-31")

    model = parser.add_argument_group("model")
    model.add_argument("--seq-len", type=int, default=20)
    model.add_argument("--market-kernel-sizes", type=int, nargs="+", default=[4, 10, 20])
    model.add_argument("--market-init-sparsity", type=float, default=0.2)
    model.add_argument("--gat-hidden-channels", type=int, default=32)
    model.add_argument(
        "--gat-out-channels",
        type=str,
        default="auto",
        help="Use 'auto' to match the input feature dimension (legacy behavior).",
    )
    model.add_argument("--gat-layers", type=int, default=2)
    model.add_argument("--gat-heads", type=int, default=2)
    model.add_argument("--mamba-hidden-sizes", type=int, nargs="+", default=[64, 64])
    model.add_argument("--mamba-num-heads", type=int, default=2)
    model.add_argument("--mamba-output-size", type=int, default=16)
    model.add_argument("--mamba-d-state", type=int, default=128)
    model.add_argument("--mamba-d-conv", type=int, default=2)
    model.add_argument("--mamba-expand", type=int, default=1)
    model.add_argument("--dropout", type=float, default=0.1)

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=5)
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--learning-rate", type=float, default=0.005)
    training.add_argument("--weight-decay", type=float, default=1e-7)
    training.add_argument("--hinge-weight", type=float, default=3.0)
    training.add_argument("--mse-weight", type=float, default=1.0)
    training.add_argument("--patience", type=int, default=10)
    training.add_argument("--log-interval", type=int, default=10)
    training.add_argument("--seed", type=int, default=2024)
    training.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, cuda:0, cuda:1, ...",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=Path, default=Path("."))
    output.add_argument("--checkpoint-name", type=str, default="best_model.pth")
    output.add_argument("--scores-name", type=str, default="scores.csv")
    output.add_argument("--prediction-name", type=str, default="pred.csv")
    output.add_argument(
        "--prediction-layout",
        choices=("legacy", "date-major"),
        default="legacy",
        help=(
            "legacy reproduces the original stock-major CSV assignment; "
            "date-major assigns scores in tensor row order."
        ),
    )

    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()
