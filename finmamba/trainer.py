from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import FinMambaData, RelationStore, TensorSplit

if TYPE_CHECKING:
    from .models import FinMamba


class HingeMSELoss(nn.Module):
    """Pointwise MSE plus the pairwise ranking hinge loss used by FinMamba."""

    def __init__(
        self,
        *,
        hinge_weight: float = 1.0,
        mse_weight: float = 1.0,
        stock_num: int = 100,
    ) -> None:
        super().__init__()
        if stock_num <= 0:
            raise ValueError(f"stock_num must be positive, got {stock_num}.")
        self.hinge_weight = hinge_weight
        self.mse_weight = mse_weight
        self.mse_loss = nn.MSELoss()
        self.stock_num = stock_num

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if scores.shape != targets.shape:
            raise ValueError(
                f"scores and targets must share a shape, got {tuple(scores.shape)} and "
                f"{tuple(targets.shape)}"
            )
        mse_loss = self.mse_loss(scores, targets)
        scores_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
        targets_diff = targets.unsqueeze(2) - targets.unsqueeze(1)
        hinge_loss = F.relu(-(scores_diff * targets_diff))
        hinge_loss = hinge_loss.sum(dim=[1, 2]) / self.stock_num
        hinge_loss = hinge_loss.mean()
        return self.hinge_weight * hinge_loss + self.mse_weight * mse_loss


@dataclass(frozen=True)
class FitResult:
    best_validation_loss: float
    epochs_completed: int


def _duplicate_feature_window(features: torch.Tensor, indices: range) -> list[torch.Tensor]:
    return [torch.cat((features[index], features[index]), dim=1) for index in indices]


def _check_sequence_capacity(data: FinMambaData, seq_len: int) -> None:
    warmup = seq_len - 1
    if data.train.num_days < warmup:
        raise ValueError(
            f"Training split has {data.train.num_days} days, but seq_len={seq_len} requires "
            f"at least {warmup} warm-up days."
        )
    if data.valid.num_days < warmup:
        raise ValueError(
            f"Validation split has {data.valid.num_days} days, but test warm-up requires "
            f"at least {warmup} days."
        )


def train_one_epoch(
    *,
    model: FinMamba,
    data: FinMambaData,
    relations: RelationStore,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    seq_len: int,
    batch_size: int,
    log_interval: int,
) -> float:
    model.train()
    window = _duplicate_feature_window(data.train.features, range(seq_len - 1))

    epoch_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for batch_start in range(seq_len - 1, data.train.num_days, batch_size):
        loss_gib: torch.Tensor | float = 0.0
        scores: list[torch.Tensor] = []
        batch_end = min(batch_start + batch_size, data.train.num_days)

        for day_index in range(batch_start, batch_end):
            if len(window) != seq_len - 1:
                raise RuntimeError(
                    f"Window has {len(window)} entries before day {day_index}; "
                    f"expected {seq_len - 1}."
                )
            relation = relations.load(day_index)
            score, window, day_gib = model(
                data.train.features[day_index],
                relation,
                data.market[day_index - seq_len + 1 : day_index + 1],
                window,
                is_training=True,
            )
            loss_gib = loss_gib + day_gib
            scores.append(score.unsqueeze(0))

        batch_scores = torch.cat(scores, dim=0)
        count = batch_end - batch_start
        loss = (
            criterion(batch_scores, data.train.labels[batch_start:batch_end])
            + loss_gib / count
        )
        epoch_loss += loss.item()

        if log_interval > 0 and n_batches % log_interval == 0:
            print(f"batch{n_batches} loss: {loss.item()}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Preserve the original truncated-BPTT boundary between batches.
        window = [entry.detach() for entry in window]
        n_batches += 1

    if n_batches == 0:
        raise ValueError("No training batches were produced. Check split dates and seq_len.")
    return epoch_loss / n_batches


def predict_split(
    *,
    model: FinMamba,
    split: TensorSplit,
    warmup_features: torch.Tensor,
    global_day_offset: int,
    market: torch.Tensor,
    relations: RelationStore,
    seq_len: int,
    is_training: bool,
) -> torch.Tensor:
    warmup_start = warmup_features.size(0) - (seq_len - 1)
    if warmup_start < 0:
        raise ValueError(
            f"Warm-up source has {warmup_features.size(0)} days; seq_len={seq_len} needs "
            f"{seq_len - 1}."
        )
    window = _duplicate_feature_window(
        warmup_features,
        range(warmup_start, warmup_features.size(0)),
    )

    scores: list[torch.Tensor] = []
    for local_day in range(split.num_days):
        global_day = global_day_offset + local_day
        if len(window) != seq_len - 1:
            raise RuntimeError(
                f"Window has {len(window)} entries before global day {global_day}; "
                f"expected {seq_len - 1}."
            )
        relation = relations.load(global_day)
        score, window, _ = model(
            split.features[local_day],
            relation,
            market[global_day - seq_len + 1 : global_day + 1],
            window,
            is_training=is_training,
        )
        scores.append(score.unsqueeze(0))
    return torch.cat(scores, dim=0)


def fit(
    *,
    model: FinMamba,
    data: FinMambaData,
    relations: RelationStore,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    seq_len: int,
    batch_size: int,
    patience: int,
    log_interval: int,
    checkpoint_path: Path,
) -> FitResult:
    _check_sequence_capacity(data, seq_len)
    best_validation_loss = float("inf")
    early_stop_counter = 0
    epochs_completed = 0

    for epoch in range(epochs):
        print(f"========== epoch {epoch} ==========")
        train_loss = train_one_epoch(
            model=model,
            data=data,
            relations=relations,
            criterion=criterion,
            optimizer=optimizer,
            seq_len=seq_len,
            batch_size=batch_size,
            log_interval=log_interval,
        )
        print(f"train loss: {train_loss}")

        model.eval()
        with torch.no_grad():
            validation_scores = predict_split(
                model=model,
                split=data.valid,
                warmup_features=data.train.features,
                global_day_offset=data.train.num_days,
                market=data.market,
                relations=relations,
                seq_len=seq_len,
                is_training=False,
            )
            validation_loss = criterion(validation_scores, data.valid.labels)

        validation_value = float(validation_loss.item())
        print(f"valid loss: {validation_value}")
        epochs_completed = epoch + 1

        if validation_value < best_validation_loss:
            best_validation_loss = validation_value
            early_stop_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    if not checkpoint_path.exists():
        raise RuntimeError(f"No checkpoint was saved to {checkpoint_path}")
    return FitResult(
        best_validation_loss=best_validation_loss,
        epochs_completed=epochs_completed,
    )


def evaluate_test(
    *,
    model: FinMamba,
    data: FinMambaData,
    relations: RelationStore,
    criterion: nn.Module,
    seq_len: int,
) -> tuple[torch.Tensor, float]:
    model.eval()
    with torch.no_grad():
        test_scores = predict_split(
            model=model,
            split=data.test,
            warmup_features=data.valid.features,
            global_day_offset=data.train.num_days + data.valid.num_days,
            market=data.market,
            relations=relations,
            seq_len=seq_len,
            is_training=False,
        )
        test_loss = criterion(test_scores, data.test.labels)
    return test_scores, float(test_loss.item())
