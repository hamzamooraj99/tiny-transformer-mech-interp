"""
eval.py

Quantitative evaluation for TinyCausalTransformer on synthetic next-token tasks.

This module focuses on task-level performance metrics (not mechanistic probes):
    - cross-entropy loss
    - next-token accuracy

It evaluates a saved checkpoint on freshly generated synthetic batches.
For induction tasks, metrics can optionally be reported over position windows
(e.g., first repeat vs later repeats) to diagnose generalization within a sequence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import wandb

from data import generate_batch
from model import TinyCausalTransformer
from config import EVAL_CFG as CFG, TRAIN_CFG, WANDB_PROJECT
from seed_utils import set_global_seed


def load_model_from_ckpt(
    ckpt_path: str,
    device: str,
    cfg: dict,
) -> TinyCausalTransformer:
    """
    Instantiate TinyCausalTransformer from cfg and load a state_dict checkpoint.
    """
    model = TinyCausalTransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        mlp_hidden_dim=cfg["mlp_hidden_dim"],
        max_seq_len=cfg["max_seq_len"],
    )
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def eval_one_batch(
    model: TinyCausalTransformer,
    seq: torch.Tensor,
    vocab_size: int,
    induction_start: int | None = None,
) -> dict:
    """
    Compute loss + accuracy for a single batch.

    seq: (B, T) token IDs
    """
    x = seq[:, :-1]
    y = seq[:, 1:]

    logits = model(x)  # (B, T-1, V)

    preds = logits.argmax(dim=-1)  # (B, T-1)
    token_losses = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),
        reduction="none",
    ).view_as(y)
    token_acc = (preds == y).float()

    metrics = {
        "loss": token_losses.mean().item(),
        "acc": token_acc.mean().item(),
    }

    if induction_start is not None:
        # y[:, i] predicts the original sequence token at position i+1.
        # To evaluate induction behavior, keep positions >= induction_start.
        start_idx = max(induction_start - 1, 0)
        pos = torch.arange(y.size(1), device=y.device)
        mask = pos >= start_idx

        if mask.any():
            metrics["induction/loss"] = token_losses[:, mask].mean().item()
            metrics["induction/acc"] = token_acc[:, mask].mean().item()

    if induction_start is not None:
        start_idx = max(induction_start - 1, 0)
        early = slice(0, start_idx)

        metrics["early/acc"] = token_acc[:, early].mean().item() if start_idx > 0 else float("nan")
        metrics["early/loss"] = token_losses[:, early].mean().item() if start_idx > 0 else float("nan")

    return metrics


@torch.no_grad()
def evaluate(
    model: TinyCausalTransformer,
    cfg: dict,
    device: str,
    n_batches: int = 50,
    mode: str = "induction",
) -> dict:
    """
    Run evaluation over n_batches newly generated batches and return mean metrics.
    """
    losses = []
    accs = []
    induction_losses = []
    induction_accs = []
    early_losses = []
    early_accs = []
    phase_shift = int(cfg.get("phase_shift", 0))

    for _ in range(n_batches):
        seq = generate_batch(
            batch_size=cfg["batch_size"],
            seq_len=cfg["seq_len"],
            vocab_size=cfg["vocab_size"],
            pattern_len=cfg["pattern_len"],
            seed=None,  # IMPORTANT: eval should sample fresh batches unless you want a fixed probe
            device=device,
            p_corrupt=cfg.get("p_corrupt_eval", 0.0),
        )

        #============ INTERVENTION 2 ============
        # if phase_shift != 0:
        #     seq = torch.roll(seq, shifts=-phase_shift, dims=1)
        #========================================

        #============ INTERVENTION 3 ============
        K = int(cfg.get("prefix_len", 0))
        if K > 0:
            prefix = torch.randint(
                0, cfg["vocab_size"],
                (seq.size(0), K),
                device=seq.device,
                dtype=seq.dtype
            )
            seq = torch.cat([prefix, seq[:, :-K]], dim=1)
        #========================================

        induction_start = cfg["pattern_len"]
        if mode == "intervention3":
            induction_start = cfg["pattern_len"] + K

        metrics = eval_one_batch(
            model=model,
            seq=seq,
            vocab_size=cfg["vocab_size"],
            induction_start=induction_start,
        )
        # --- debug prints (temporary) ---
        # if len(losses) == 0:  # first batch only
        #     print("one-batch metrics:", metrics)

        #     # optional: also print basic sequence example
        #     print("seq[0]:", seq[0].tolist())
        # --- end debug prints ---
        losses.append(metrics["loss"])
        accs.append(metrics["acc"])
        if "induction/loss" in metrics and "induction/acc" in metrics:
            induction_losses.append(metrics["induction/loss"])
            induction_accs.append(metrics["induction/acc"])
        if "early/loss" in metrics and "early/acc" in metrics:
            early_losses.append(metrics["early/loss"])
            early_accs.append(metrics["early/acc"])

    summary = {
        "eval/loss": sum(losses) / len(losses),
        "eval/acc": sum(accs) / len(accs),
        "eval/n_batches": n_batches,
    }
    if induction_losses and induction_accs:
        summary["eval/induction/loss"] = sum(induction_losses) / len(induction_losses)
        summary["eval/induction/acc"] = sum(induction_accs) / len(induction_accs)
    if early_losses and early_accs:
        summary["eval/early/loss"] = sum(early_losses) / len(early_losses)
        summary["eval/early/acc"] = sum(early_accs) / len(early_accs)
    return summary


def log_eval_bar_charts(metrics: dict) -> None:
    acc_keys = ["eval/acc", "eval/induction/acc", "eval/early/acc"]
    loss_keys = ["eval/loss", "eval/induction/loss", "eval/early/loss"]

    acc_rows = [[k, metrics[k]] for k in acc_keys if k in metrics]
    if acc_rows:
        acc_table = wandb.Table(data=acc_rows, columns=["metric", "value"])
        wandb.log(
            {
                "eval/accuracy_bar": wandb.plot.bar(
                    acc_table, "metric", "value", title="Evaluation Accuracy Metrics"
                )
            },
            step=0,
        )

    loss_rows = [[k, metrics[k]] for k in loss_keys if k in metrics]
    if loss_rows:
        loss_table = wandb.Table(data=loss_rows, columns=["metric", "value"])
        wandb.log(
            {
                "eval/loss_bar": wandb.plot.bar(
                    loss_table, "metric", "value", title="Evaluation Loss Metrics"
                )
            },
            step=0,
        )


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = CFG["exp_name"]
    CFG['run_name'] = f"Induction_{exp_name}-EVAL"
    ckpt_path = "checkpoints/baseline.pt" 
    mode = "intervention3"
    if CFG.get("seed") is not None:
        set_global_seed(CFG["seed"])

    model = load_model_from_ckpt(ckpt_path=ckpt_path, device=DEVICE, cfg=CFG)
    #============ INTERVENTION 1 ============
    # model.pos_emb.weight.data.zero_()
    #========================================

    wandb.init(
        project=WANDB_PROJECT,
        name=CFG['run_name'],
        config={**CFG, "checkpoint_path": ckpt_path, "mode": mode},
        job_type="eval",
    )

    metrics = evaluate(
        model=model,
        cfg=CFG,
        device=DEVICE,
        n_batches=CFG.get("eval_batches", 50),
        mode=mode,
    )
    for k, v in metrics.items():
        wandb.run.summary[k] = v
    log_eval_bar_charts(metrics)
    wandb.finish()
    # print(metrics)


if __name__ == "__main__":
    main()
