from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def save_score_matrix(scores: torch.Tensor, path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(scores.detach().cpu().numpy())
    frame.to_csv(path, index=False, header=False)
    return frame


def save_prediction_frame(
    *,
    label_frame: pd.DataFrame,
    scores: torch.Tensor,
    test_start: str,
    test_end: str,
    path: Path,
    layout: str,
) -> pd.DataFrame:
    score_array = scores.detach().cpu().numpy()
    datetimes = pd.to_datetime(label_frame["datetime"])

    if layout == "legacy":
        # Preserve the source exactly: lower-bound filter only and stock-major assignment,
        # equivalent to df_scores[int(i / test_day_num)][i % test_day_num].
        output = label_frame.loc[datetimes >= pd.Timestamp(test_start)].reset_index(drop=True)
        flat_scores = score_array.T.reshape(-1)
    elif layout == "date-major":
        mask = (datetimes >= pd.Timestamp(test_start)) & (
            datetimes <= pd.Timestamp(test_end)
        )
        output = label_frame.loc[mask].reset_index(drop=True)
        flat_scores = score_array.reshape(-1)
    else:  # Defensive guard for direct Python use.
        raise ValueError(f"Unknown prediction layout: {layout!r}")

    if len(output) > len(flat_scores):
        raise ValueError(
            f"Prediction CSV needs {len(output)} values, but score tensor provides only "
            f"{len(flat_scores)}. Check the test date range and panel ordering."
        )
    output = output.copy()
    output.loc[:, "label"] = flat_scores[: len(output)]
    # index=True intentionally matches lab.to_csv('pred.csv') in the original script.
    output.to_csv(path)
    return output
