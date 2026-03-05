WANDB_PROJECT = "Tiny-Transformer-Mech-Interp"

EXP_NAME = "exp_3"

TRAIN_CFG = {
    # --- data ---
    "vocab_size": 20,
    "seq_len": 32,
    "pattern_len": 8,
    "batch_size": 64,
    "max_seq_len": 32,
    'p_corrupt_train': 0.0,
    "seed": 42,
    # --- model ---
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "mlp_hidden_dim": 256,
    "lr": 1e-3,
    "steps": 2000,
    # --- log ---
    'run_name': "TinyTransformer"
}

EVAL_CFG = {
    # --- data ---
    "vocab_size": 20,
    "seq_len": 32,
    "pattern_len": 8,
    "batch_size": 64,
    "max_seq_len": 32,
    # --- model ---
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "mlp_hidden_dim": 256,
    # --- eval ---
    "seed": 3407,
    "p_corrupt_eval": 0.0,
    "exp_name": EXP_NAME,
    "eval_batches": 50,
    # "phase_shift": 3, #INTERVENTION 2
    "prefix_len": 3, #INTERVENTION 3
}

MI_CFG = {
    "vocab_size": 20,
    "seq_len": 32,
    "pattern_len": 8,
    "batch_size": 1,     # for mech interp, start with B=1
    "max_seq_len": 32,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "mlp_hidden_dim": 256,
    "seed_probe": 6532,
    "p_corrupt_probe": 0.0,
    "exp_name": EXP_NAME,
    # "phase_shift": 3, #INTERVENTION 2
    "prefix_len": 3, #INTERVENTION 3
}
