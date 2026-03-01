

import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from src.data import generate_batch
from src.model import TinyCausalTransformer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CFG = {
    'vocab_size': 20,
    'seq_len': 32,
    'pattern_len': 8,
    'batch_size': 64,
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 2,
    'mlp_hidden_dim': 256,
    'max_seq_len': 32,
    'lr': 1e-3,
    'steps': 2000,
    'p_corrupt_train': 0.0,
    'seed': 42,
    'run_name': "TinyTransformerTest1"
}

wandb.init(
    project="Tiny-Transformer-Mech-Interp",
    name=CFG['run_name'],
    config=CFG,
)

if __name__ == "__main__":
    model = TinyCausalTransformer(
        vocab_size=CFG['vocab_size'],
        d_model=CFG['d_model'],
        n_heads=CFG['n_heads'],
        n_layers=CFG['n_layers'],
        mlp_hidden_dim=CFG['mlp_hidden_dim'],
        max_seq_len=CFG['max_seq_len'],
    )
    model.to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=CFG['lr'])
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    # print("n_params:", n_params)

    pbar = tqdm(range(CFG["steps"]))
    for step in pbar:   
        #1. Generate Batch
        seq = generate_batch(
            batch_size=CFG['batch_size'],
            seq_len=CFG['seq_len'],
            vocab_size=CFG['vocab_size'],
            pattern_len=CFG['pattern_len'],
            seed=CFG['seed'],
            device=DEVICE,
            p_corrupt=CFG['p_corrupt_train'],
            mode="induction"
        )

        #2. Separate x & y
        x = seq[:, :-1]
        y = seq[:, 1:]

        #3. Forward
        logits = model(x)

        #4. Compute Loss
        loss = F.cross_entropy(
            logits.reshape(-1, CFG['vocab_size']),
            y.reshape(-1)
        )

        #5. Zero Grad
        optim.zero_grad()

        #6. Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        #7. Optimiser Step
        optim.step()

        #8. Log
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        if step % 50 == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/step": step
            })

    wandb.finish()