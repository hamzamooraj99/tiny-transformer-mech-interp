"""
viz.py

Visualization utilities for mechanistic interpretability outputs.

This file is intentionally separate from mech_interp.py to keep:
    - probes (numbers) and
    - visualizations (plots)
decoupled.

Typical visualizations:
    - attention heatmap for a specific layer/head and query range
    - bar plot of argmax frequencies by key position
    - line/bar plots of probe scores across heads/layers
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
import numpy as np
import json


def plot_attention_heatmap(
    attn: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    pattern_len: int | None = None,
    title: str | None = None,
    max_tokens: int | None = None,
):
    """
    Plot a (T, T) attention matrix heatmap for a specific layer/head.

    Args:
        attn:
            Attention tensor of shape (B, H, T, T) or (H, T, T) or (T, T).
        layer_idx, head_idx:
            Which layer/head is being visualized (for labeling).
        max_tokens:
            Optionally crop to top-left (max_tokens x max_tokens) for readability.
    """
    # --- normalize to a (T, T) matrix A ---
    if attn.dim() == 4:
        # (B, H, T, T)
        A = attn[0, head_idx]
    elif attn.dim() == 3:
        # (H, T, T)
        A = attn[head_idx]
    elif attn.dim() == 2:
        # (T, T)
        A = attn
    else:
        raise ValueError(f"Unexpected attn shape: {tuple(attn.shape)}")

    # Optional crop for readability
    if max_tokens is not None:
        A = A[:max_tokens, :max_tokens]

    # Move to CPU numpy for matplotlib
    A_np = A.detach().cpu().numpy()

    T = A_np.shape[0]

    # --- plot ---
    plt.figure()
    plt.imshow(A_np, aspect="auto")
    plt.colorbar()

    # Overlay expected induction diagonal j = i - P
    if pattern_len is not None and pattern_len >= 0:
        xs = []
        ys = []
        for i in range(pattern_len, T):
            j = i - pattern_len
            # Keep overlay in visible bounds (important when max_tokens crops A)
            if 0 <= j < T:
                xs.append(j)
                ys.append(i)

        if xs:
            plt.plot(
                xs,
                ys,
                linestyle="--",
                linewidth=2,
                color="white",
                label="expected induction (j = i - P)",
            )
            plt.legend()

    if title is None:
        title = f"Attention heatmap | layer={layer_idx} head={head_idx}"
    plt.title(title)
    plt.xlabel("Key position (j)")
    plt.ylabel("Query position (i)")


def plot_argmax_histogram(
    counts: torch.Tensor,
    title: str | None = None,
):
    """
    Bar plot of argmax counts per key position.

    counts: (T,) integer counts where counts[j] = number of rows whose argmax key is j
    """
    if counts.dim() != 1:
        raise ValueError(f"counts must be 1D, got shape: {tuple(counts.shape)}")

    c = counts.detach().cpu().numpy()
    xs = list(range(len(c)))

    plt.figure()
    plt.bar(xs, c)

    if title is None:
        title = "Argmax key-position counts"
    plt.title(title)
    plt.xlabel("Key position (j)")
    plt.ylabel("#Rows where argmax key == j")


def plot_head_scores(
    head_scores: list[float],
    title: str | None = None,
):
    """
    Simple bar plot of scores for heads within one layer.

    head_scores: list length H
    """
    xs = list(range(len(head_scores)))

    plt.figure()
    plt.bar(xs, head_scores)

    if title is None:
        title = "Head scores"
    plt.title(title)
    plt.xlabel("Head index")
    plt.ylabel("Score")

def plot_offset_diagonal_mean_std(
    means_by_layer: list[list[float]],
    stds_by_layer: list[list[float]],
    title: str | None = None,
    layer_labels: list[str] | None = None,
):
    """
    Grouped bar chart for offset-diagonal attention: mean ± std across probe seeds.

    Args:
        means_by_layer: list length L, each is list length H (means per head)
        stds_by_layer:  list length L, each is list length H (std per head)
    """
    L = len(means_by_layer)
    H = len(means_by_layer[0])

    means = np.array(means_by_layer)  # (L, H)
    stds = np.array(stds_by_layer)    # (L, H)

    x = np.arange(H)
    width = 0.8 / L  # keep bars within one head slot

    plt.figure()
    for layer_idx in range(L):
        offset = (layer_idx - (L - 1) / 2) * width
        lbl = layer_labels[layer_idx] if layer_labels is not None else f"Layer {layer_idx}"
        plt.bar(
            x + offset,
            means[layer_idx],
            width=width,
            yerr=stds[layer_idx],
            capsize=3,
            label=lbl,
        )

    plt.xticks(x, [f"H{h}" for h in range(H)])
    if title is None:
        title = "Offset-diagonal attention (mean ± std)"
    plt.title(title)
    plt.xlabel("Head")
    plt.ylabel("Attention weight on j = i - P")
    plt.legend()

if __name__ == "__main__":
    import os

    exp_name = "exp_3"  # update if needed
    viz_seed = 0        # should match saved folder name

    viz_dir = f"mech_viz/{exp_name}_viz_seed_{viz_seed}"
    out_dir = os.path.join(viz_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    seq = torch.load(os.path.join(viz_dir, "seq.pt"))
    attn_all = torch.load(os.path.join(viz_dir, "attn_all.pt"))  # list of tensors

    print("seq shape:", tuple(seq.shape))
    print("n_layers:", len(attn_all), "attn[0] shape:", tuple(attn_all[0].shape))

    # ---- 1) Heatmaps: layer x head ----
    for layer_idx, attn_layer in enumerate(attn_all):
        H = attn_layer.shape[1]
        for head_idx in range(H):
            plot_attention_heatmap(
                attn=attn_layer,
                layer_idx=layer_idx,
                head_idx=head_idx,
                pattern_len=8,
                max_tokens=31,
            )
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"attn_heatmap_L{layer_idx}_H{head_idx}.png"), dpi=200)
            plt.close()

    # ---- 2) Argmax histograms: layer x head ----
    P = 8  # pattern length; keep consistent with config
    for layer_idx, attn_layer in enumerate(attn_all):
        H = attn_layer.shape[1]
        T = attn_layer.shape[-1]
        start_i = P

        for head_idx in range(H):
            counts = torch.zeros(T, dtype=torch.long)
            for i in range(start_i, T):
                row = attn_layer[0, head_idx, i]  # (T,)
                k = int(row.argmax().item())
                counts[k] += 1

            plot_argmax_histogram(
                counts=counts,
                title=f"Argmax histogram | layer={layer_idx} head={head_idx}",
            )
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"argmax_hist_L{layer_idx}_H{head_idx}.png"), dpi=200)
            plt.close()
    
    # ---- 3) Offset-diagonal mean±std (from aggregated JSON) ----
    agg_path = f"artifacts/{exp_name}/probes_agg.json"
    with open(agg_path, "r") as f:
        agg = json.load(f)

    P = agg["offset_diag"]["meta"]["pattern_len"] if "meta" in agg["offset_diag"] else agg["meta"]["pattern_len"]
    # (Depending on your JSON structure, pattern_len might live in agg["meta"] only.
    # If the line above errors, just do: P = agg["meta"]["pattern_len"])

    # Extract means/stds for each layer
    # Keys look like: "layer_0/offset_minus_8/mean"
    means_by_layer = []
    stds_by_layer = []

    # infer number of layers by scanning keys
    offset_keys = list(agg["offset_diag"].keys())
    layer_ids = sorted({k.split("/")[0] for k in offset_keys if k.startswith("layer_")})
    for layer_id in layer_ids:
        mean_key = f"{layer_id}/offset_minus_{agg['meta']['pattern_len']}/mean"
        std_key  = f"{layer_id}/offset_minus_{agg['meta']['pattern_len']}/std"
        means_by_layer.append(agg["offset_diag"][mean_key])
        stds_by_layer.append(agg["offset_diag"][std_key])

    plot_offset_diagonal_mean_std(
        means_by_layer=means_by_layer,
        stds_by_layer=stds_by_layer,
        title=f"Offset-diagonal attention (P={agg['meta']['pattern_len']})",
        layer_labels=[lid.replace("layer_", "Layer ") for lid in layer_ids],
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "offset_diagonal_mean_std.png"), dpi=200)
    plt.close()

    print("Saved figures to:", out_dir)
