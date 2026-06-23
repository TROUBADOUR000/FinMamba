from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


_COLUMN_ALIASES = {
    "kdcode": "instrument",
    "dt": "datetime",
    "Company": "instrument",
    "Date": "datetime",
}


@dataclass(frozen=True)
class TensorSplit:
    features: torch.Tensor
    labels: torch.Tensor
    num_days: int


@dataclass(frozen=True)
class FinMambaData:
    train: TensorSplit
    valid: TensorSplit
    test: TensorSplit
    market: torch.Tensor
    industry_relation: torch.Tensor
    stock_num: int
    feature_dim: int
    feature_frame: pd.DataFrame
    label_frame: pd.DataFrame


def normalize_panel_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Reset the index and normalize the two metadata column names."""
    normalized = frame.reset_index()
    normalized = normalized.rename(columns=_COLUMN_ALIASES)
    missing = {"instrument", "datetime"} - set(normalized.columns)
    if missing:
        raise ValueError(
            "Panel data must contain instrument/datetime columns (or supported aliases); "
            f"missing: {sorted(missing)}"
        )
    return normalized


def _date_mask(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    datetimes = pd.to_datetime(frame["datetime"])
    return (datetimes >= pd.Timestamp(start)) & (datetimes <= pd.Timestamp(end))


def _warn_if_not_date_major(frame: pd.DataFrame, stock_num: int, split_name: str) -> None:
    """Warn only; the reshape still follows the original script's row order."""
    if frame.empty or stock_num <= 0 or len(frame) % stock_num != 0:
        return
    datetimes = frame["datetime"].to_numpy()
    blocks = datetimes.reshape(-1, stock_num)
    if any(len(set(block.tolist())) != 1 for block in blocks):
        warnings.warn(
            f"{split_name} rows are not grouped into one datetime per contiguous stock block. "
            "The refactor intentionally preserves the original reshape order, so verify the "
            "input pickle ordering."
        )


def _reshape_features(
    frame: pd.DataFrame,
    *,
    stock_num: int,
    feature_dim: int,
    split_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    day_num = int(frame["datetime"].nunique())
    values = frame.drop(columns=["datetime", "instrument"]).to_numpy()
    expected_rows = day_num * stock_num
    if values.shape != (expected_rows, feature_dim):
        raise ValueError(
            f"{split_name} feature shape is {values.shape}, expected "
            f"({expected_rows}, {feature_dim}) = days({day_num}) * stocks({stock_num})."
        )
    _warn_if_not_date_major(frame, stock_num, split_name)
    array = values.reshape(day_num, stock_num, feature_dim)
    return torch.as_tensor(array, dtype=torch.float32, device=device), day_num


def _reshape_labels(
    frame: pd.DataFrame,
    *,
    stock_num: int,
    day_num: int,
    split_name: str,
    device: torch.device,
) -> torch.Tensor:
    values = frame.drop(columns=["datetime", "instrument"]).to_numpy()
    expected_rows = day_num * stock_num
    if values.shape != (expected_rows, 1):
        raise ValueError(
            f"{split_name} label shape is {values.shape}, expected ({expected_rows}, 1). "
            "The legacy code requires exactly one label column."
        )
    _warn_if_not_date_major(frame, stock_num, f"{split_name} labels")
    array = values.reshape(day_num, stock_num)
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def load_finmamba_data(
    *,
    stock: str,
    data_dir: Path,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
    device: torch.device,
) -> FinMambaData:
    feature_path = data_dir / f"{stock}fea.pkl"
    label_path = data_dir / f"{stock}lab.pkl"
    industry_path = data_dir / f"{stock}_industry_relationship.npy"

    for path in (feature_path, label_path, industry_path):
        if not path.exists():
            raise FileNotFoundError(f"Required data file not found: {path}")

    feature_frame = normalize_panel_columns(pd.read_pickle(feature_path))
    label_frame = normalize_panel_columns(pd.read_pickle(label_path))

    stock_num = int(feature_frame["instrument"].nunique())
    feature_dim = int(feature_frame.shape[1] - 2)
    if stock_num <= 0 or feature_dim <= 0:
        raise ValueError(
            f"Invalid panel dimensions: stock_num={stock_num}, feature_dim={feature_dim}."
        )

    split_ranges = {
        "train": (train_start, train_end),
        "valid": (valid_start, valid_end),
        "test": (test_start, test_end),
    }
    feature_splits: dict[str, pd.DataFrame] = {}
    label_splits: dict[str, pd.DataFrame] = {}
    for name, (start, end) in split_ranges.items():
        feature_splits[name] = feature_frame.loc[_date_mask(feature_frame, start, end)]
        label_splits[name] = label_frame.loc[_date_mask(label_frame, start, end)]
        if feature_splits[name].empty:
            raise ValueError(f"The {name} feature split ({start} to {end}) is empty.")
        if label_splits[name].empty:
            raise ValueError(f"The {name} label split ({start} to {end}) is empty.")

    split_tensors: dict[str, TensorSplit] = {}
    for name in ("train", "valid", "test"):
        features, day_num = _reshape_features(
            feature_splits[name],
            stock_num=stock_num,
            feature_dim=feature_dim,
            split_name=name,
            device=device,
        )
        labels = _reshape_labels(
            label_splits[name],
            stock_num=stock_num,
            day_num=day_num,
            split_name=name,
            device=device,
        )
        split_tensors[name] = TensorSplit(features=features, labels=labels, num_days=day_num)

    market = torch.cat(
        [
            split_tensors["train"].features.mean(dim=1),
            split_tensors["valid"].features.mean(dim=1),
            split_tensors["test"].features.mean(dim=1),
        ],
        dim=0,
    )

    industry_array = np.load(industry_path)
    if industry_array.shape != (stock_num, stock_num):
        raise ValueError(
            f"Industry relation shape is {industry_array.shape}, expected "
            f"({stock_num}, {stock_num})."
        )
    # Deliberately preserve NumPy's dtype, matching torch.tensor(p) in the legacy code.
    industry_relation = torch.tensor(industry_array, device=device)

    return FinMambaData(
        train=split_tensors["train"],
        valid=split_tensors["valid"],
        test=split_tensors["test"],
        market=market,
        industry_relation=industry_relation,
        stock_num=stock_num,
        feature_dim=feature_dim,
        feature_frame=feature_frame,
        label_frame=label_frame,
    )


class RelationStore:
    """Load one dynamic stock relation matrix by its global day index."""

    def __init__(
        self,
        *,
        relation_dir: Path,
        relation_pattern: str,
        industry_relation: torch.Tensor,
        stock_num: int,
        device: torch.device,
    ) -> None:
        self.relation_dir = relation_dir
        self.relation_pattern = relation_pattern
        self.industry_relation = industry_relation
        self.stock_num = stock_num
        self.device = device

    def path_for(self, day_index: int) -> Path:
        try:
            filename = self.relation_pattern.format(index=day_index, day=day_index)
        except (KeyError, IndexError) as exc:
            raise ValueError(
                "relation-pattern must use {index} or {day}, for example day{index}.pkl"
            ) from exc
        return self.relation_dir / filename

    def load(self, day_index: int) -> torch.Tensor:
        path = self.path_for(day_index)
        if not path.exists():
            raise FileNotFoundError(f"Relation file not found for day {day_index}: {path}")
        with path.open("rb") as handle:
            relation: Any = pickle.load(handle)
        relation_tensor = torch.as_tensor(relation, dtype=torch.float32, device=self.device)
        if relation_tensor.shape != (self.stock_num, self.stock_num):
            raise ValueError(
                f"Relation matrix {path} has shape {tuple(relation_tensor.shape)}, expected "
                f"({self.stock_num}, {self.stock_num})."
            )
        return self.industry_relation * relation_tensor
