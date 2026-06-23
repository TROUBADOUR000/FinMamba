from __future__ import annotations

import torch

from finmamba.config import parse_args
from finmamba.data import RelationStore, load_finmamba_data
from finmamba.trainer import HingeMSELoss, evaluate_test, fit
from finmamba.models import FinMamba
from finmamba.outputs import save_prediction_frame, save_score_matrix
from finmamba.utils import ensure_directory, resolve_device, set_seed


def _resolve_out_channels(value: str, feature_dim: int) -> int:
    if value.lower() == "auto":
        return feature_dim
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"--gat-out-channels must be 'auto' or an integer, got {value!r}"
        ) from exc


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_directory(args.output_dir)

    print(f"device: {device}")
    data = load_finmamba_data(
        stock=args.stock,
        data_dir=args.data_dir,
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        test_start=args.test_start,
        test_end=args.test_end,
        device=device,
    )
    print(
        "data: "
        f"stocks={data.stock_num}, features={data.feature_dim}, "
        f"train_days={data.train.num_days}, valid_days={data.valid.num_days}, "
        f"test_days={data.test.num_days}"
    )

    out_channels = _resolve_out_channels(args.gat_out_channels, data.feature_dim)
    model = FinMamba(
        input_dim=data.feature_dim,
        stock_num=data.stock_num,
        hidden_channels=args.gat_hidden_channels,
        out_channels=out_channels,
        gat_layers=args.gat_layers,
        gat_heads=args.gat_heads,
        mamba_hidden_sizes=args.mamba_hidden_sizes,
        mamba_output_size=args.mamba_output_size,
        mamba_num_heads=args.mamba_num_heads,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        market_kernel_sizes=args.market_kernel_sizes,
        market_init_sparsity=args.market_init_sparsity,
        dropout=args.dropout,
        seq_len=args.seq_len,
    ).to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"trainable parameters: {trainable_parameters}")

    criterion = HingeMSELoss(
        hinge_weight=args.hinge_weight,
        mse_weight=args.mse_weight,
        stock_num=data.stock_num,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    relations = RelationStore(
        relation_dir=args.relation_dir,
        relation_pattern=args.relation_pattern,
        industry_relation=data.industry_relation,
        stock_num=data.stock_num,
        device=device,
    )

    checkpoint_path = output_dir / args.checkpoint_name
    fit_result = fit(
        model=model,
        data=data,
        relations=relations,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        patience=args.patience,
        log_interval=args.log_interval,
        checkpoint_path=checkpoint_path,
    )
    print(
        f"best valid loss: {fit_result.best_validation_loss} "
        f"after {fit_result.epochs_completed} epoch(s)"
    )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    test_scores, test_loss = evaluate_test(
        model=model,
        data=data,
        relations=relations,
        criterion=criterion,
        seq_len=args.seq_len,
    )
    print(test_scores.shape)
    print(f"test loss: {test_loss}")

    scores_path = output_dir / args.scores_name
    prediction_path = output_dir / args.prediction_name
    save_score_matrix(test_scores, scores_path)
    prediction_frame = save_prediction_frame(
        label_frame=data.label_frame,
        scores=test_scores,
        test_start=args.test_start,
        test_end=args.test_end,
        path=prediction_path,
        layout=args.prediction_layout,
    )
    print(prediction_frame.shape)
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved scores: {scores_path}")
    print(f"saved predictions: {prediction_path}")
    print("finished!")


if __name__ == "__main__":
    main()
