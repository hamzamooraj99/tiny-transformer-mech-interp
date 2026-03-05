"""
mech_interp.py

Mechanistic interpretability probes for TinyCausalTransformer on synthetic tasks.

This module focuses on hypothesis-driven analysis of internal behavior,
primarily using attention weights returned by the model.

Typical probes:
    - diagonal (offset) attention score: avg attn[i, i-P]
    - anchor-column score: avg attn[i, anchor_pos]
    - argmax frequency: how often each key position is the argmax across i
    - attention entropy / sharpness (optional)

All probes are designed to be:
    - lightweight (single batch or few batches)
    - easy to reproduce (fixed seed)
    - easy to compare across checkpoints/experiments
"""

from __future__ import annotations

import torch
import os
import json
from datetime import datetime

from data import generate_batch
from model import TinyCausalTransformer
from config import MI_CFG as CFG

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def to_jsonable(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    return str(x)

def load_model_from_ckpt(
    ckpt_path: str,
    device: str,
    cfg: dict,
) -> TinyCausalTransformer:
    """
    Instantiate model from cfg and load checkpoint. Must support return_attn=True in forward.
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
def get_attn_for_sequence(
    model: TinyCausalTransformer,
    seq: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Run the model on seq inputs (shifted) and return logits + attentions.

    Returns:
        logits: (B, T-1, V)
        attn_all: list length n_layers, each (B, H, T-1, T-1)
    """
    x = seq[:, :-1]
    logits, attn_all = model(x, return_attn=True)
    return logits, attn_all


def probe_offset_diagonal(
    attn_all: list[torch.Tensor],
    pattern_len: int,
    start_i: int | None = None,
) -> dict:
    """
    Compute avg attention on the diagonal j = i - pattern_len, per layer/head.

    This is a targeted induction-head detector when the task structure implies
    the relevant previous occurrence is exactly pattern_len steps back.
    """
    results = {}

    for layer_idx, attn in enumerate(attn_all):
        # attn: (B, H, T, T)
        B, H, T, _ = attn.shape
        P = pattern_len

        i0 = P if start_i is None else max(start_i, P)

        scores = []
        for h in range(H):
            vals = []
            for i in range(i0, T):
                vals.append(attn[0, h, i, i - P].item())
            scores.append(sum(vals) / len(vals))

        results[f"layer_{layer_idx}/offset_minus_{P}"] = scores

    return results


def probe_anchor_column(
    attn_all: list[torch.Tensor],
    key_pos: int,
    start_i: int,
) -> dict:
    """
    Compute avg attention to a fixed key_pos across query positions i>=start_i, per layer/head.

    Useful for detecting vertical-stripe / anchor behavior (e.g., always attend to position 1).
    """
    results = {}

    for layer_idx, attn in enumerate(attn_all):
        B, H, T, _ = attn.shape

        scores = []
        for h in range(H):
            vals = []
            for i in range(start_i, T):
                vals.append(attn[0, h, i, key_pos].item())
            scores.append(sum(vals) / len(vals))

        results[f"layer_{layer_idx}/key_{key_pos}_avg_from_i_{start_i}"] = scores

    return results


def probe_argmax_frequency(
    attn_all: list[torch.Tensor],
    start_i: int,
) -> dict:
    """
    For each layer/head, count how often each key position is the argmax across query rows.
    Returns a compact summary: (most_common_key, fraction).
    """
    results = {}

    for layer_idx, attn in enumerate(attn_all):
        B, H, T, _ = attn.shape

        head_summaries = []
        for h in range(H):
            counts = torch.zeros(T, dtype=torch.long)

            for i in range(start_i, T):
                row = attn[0, h, i]               # (T,)
                k = int(row.argmax().item())      # key position
                counts[k] += 1

            top_key = int(counts.argmax().item())
            frac = float(counts[top_key].item() / counts.sum().item())

            head_summaries.append({"top_key": top_key, "frac": frac})

        results[f"layer_{layer_idx}/argmax_summary_from_i_{start_i}"] = head_summaries

    return results

@torch.no_grad()
def aggregate_probes(
    model: TinyCausalTransformer,
    device: str,
    cfg: dict,
    probe_seeds: list[int],
    viz_seed: int | None=None,
    viz_dir: str | None=None
) -> dict:
    """
    Run probes across multiple fixed-seed sequences and aggregate statistics.

    Aggregation:
      - Offset-diagonal: compute per-seed per-head diagonal means, then report mean/std over seeds.
      - Argmax frequency: accumulate argmax counts across all (seed, query_row) pairs, then report top_key and fraction.
    """
    P = cfg["pattern_len"]
    phase_shift = cfg.get("phase_shift", 0)

    #Infer T from first probe once attn tensors are retrived
    diag_per_seed = [] #List of Dicts: [{'layer0': [h0,h1,h2,h3], 'layer1':[...]}, {...}, ]
    argmax_counts = None #LongTensor
    total_rows = 0

    for s in probe_seeds:
        #1) Generate a fixed probe sequence (B=1) for this seed
        seq = generate_batch(
            batch_size=1,
            seq_len=cfg["seq_len"],
            vocab_size=cfg["vocab_size"],
            pattern_len=cfg["pattern_len"],
            seed=s,
            device=device,
            p_corrupt=cfg["p_corrupt_probe"],
            mode="induction",
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

        #2) Get attn tensors for this sequence
        _, attn_all = get_attn_for_sequence(model=model, seq=seq)
        if viz_seed is not None and viz_dir is not None and s==viz_seed:
            create_viz_ckpt(viz_seed=viz_seed, viz_dir=viz_dir, cfg=cfg, attn_all=attn_all, seq=seq)
        n_layers = len(attn_all)
        _, n_heads, T, _ = attn_all[0].shape
        start_i = P if K == 0 else P + K
        rows_this_seed = (T - start_i)
        total_rows += rows_this_seed

        #Init argmax accumulator
        if argmax_counts is None:
            argmax_counts = torch.zeros((n_layers, n_heads, T), dtype=torch.long)
        
        #3) Per-seed offset-diagonal means (per layer/head)
        seed_diag = {}
        for layer_idx, attn in enumerate(attn_all):
            #attn: (1,H,T,T)
            head_means = []
            for h in range(n_heads):
                vals = []
                for i in range(start_i, T):
                    vals.append(attn[0, h, i, i-P].item())
                head_means.append(sum(vals) / len(vals))
            seed_diag[f"layer_{layer_idx}"] = head_means
        diag_per_seed.append(seed_diag)

        #4) Argmax counts across query rows (per layer/head)
        for layer_idx, attn in enumerate(attn_all):
            for h in range(n_heads):
                for i in range(start_i, T):
                    row = attn[0, h, i]
                    k = int(row.argmax().item())
                    argmax_counts[layer_idx, h, k] += 1
    
    #---- Aggregate Results ----
    #Offset-diagonal: mean/std over seeds
    offset_out = {}
    for layer_idx in range(len(diag_per_seed[0].keys())):
        key = f"layer_{layer_idx}"
        #Collect shape: (S, H)
        mat = torch.tensor([d[key] for d in diag_per_seed], dtype=torch.float32)
        offset_out[f"layer_{layer_idx}/offset_minus_{P}/mean"] = mat.mean(dim=0).tolist()
        offset_out[f"layer_{layer_idx}/offset_minus_{P}/std"] = mat.std(dim=0, unbiased=False).tolist()
    
    argmax_out = {}
    n_layers, n_heads, T = argmax_counts.shape
    for layer_idx in range(n_layers):
        summaries = []
        for h in range(n_heads):
            counts = argmax_counts[layer_idx, h]
            top_key = int(counts.argmax().item())
            frac = float(counts[top_key].item() / counts.sum().item()) if counts.sum().item() > 0 else 0.0
            summaries.append({'top_key': top_key, 'frac': frac})
        argmax_out[f"layer_{layer_idx}/argmax_summary_from_i_{start_i}"] = summaries
    
    return {
        "meta": {
            "probe_seeds": probe_seeds,
            "n_probes": len(probe_seeds),
            "pattern_len": P,
            "attn_T": int(T),
            "rows_per_probe": int(T - P),
            "total_rows": int(total_rows)
        },
        "offset_diag": offset_out,
        "argmax": argmax_out
    }

def create_viz_ckpt(
        viz_seed: int,
        viz_dir: str,
        cfg: dict,
        attn_all: list[torch.Tensor],
        seq: torch.Tensor,
):
    ensure_dir(viz_dir)
    torch.save(seq.detach().cpu(), os.path.join(viz_dir, "seq.pt"))
    torch.save([a.detach().cpu() for a in attn_all], os.path.join(viz_dir, "attn_all.pt"))

    # Optional: single-seed probe summary for captions/sanity checks
    P = cfg["pattern_len"]
    out_single = {}
    out_single.update(probe_offset_diagonal(attn_all=attn_all, pattern_len=P))
    out_single.update(probe_argmax_frequency(attn_all=attn_all, start_i=P))

    with open(os.path.join(viz_dir, "probes_single.json"), "w") as f:
        json.dump(
            to_jsonable({
                "meta": {
                    "viz_seed": viz_seed,
                    "pattern_len": P,
                    "attn_shape": list(attn_all[0].shape),
                },
                "probes_single": out_single,
            }),
            f,
            indent=2,
        )

def main():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = f"checkpoints/baseline.pt"  # TODO: make CLI arg later

    model = load_model_from_ckpt(ckpt_path=ckpt_path, device=DEVICE, cfg=CFG)
    #============ INTERVENTION 1 ============
    # model.pos_emb.weight.data.zero_()
    #========================================

    #Choose probe seeds (10 sequences)
    probe_seeds = list(range(10))
    viz_seed = probe_seeds[0]
    viz_dir = f"mech_viz/{CFG['exp_name']}_viz_seed_{viz_seed}"

    #Run aggregated probes
    agg = aggregate_probes(model=model, device=DEVICE, cfg=CFG, probe_seeds=probe_seeds, viz_seed=viz_seed, viz_dir=viz_dir)

    #Save Artifacts
    artifact_dir = f"artifacts/{CFG['exp_name']}"
    ensure_dir(artifact_dir)
    
    payload = {
        "meta": {
            "ckpt_path": ckpt_path,
            "device": DEVICE,
            "timestamp": datetime.now().isoformat(timespec="seconds")
        },
        "cfg": CFG,
        **agg,
    }
    
    out_path = os.path.join(artifact_dir, "probes_agg.json")
    with open(out_path, "w") as f:
        json.dump(to_jsonable(payload), f, indent=2)
    
    #Print a short terminal summary
    print("Saved:", out_path)
    print("Offset diagonal means/stds:")
    for k, v in payload["offset_diag"].items():
        print(k, "=>", v)
    print("Argmax summaries:")
    for k, v in payload["argmax"].items():
        print(k, "=>", v)


if __name__ == "__main__":
    main()