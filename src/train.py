

import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from data import generate_batch
from model import TinyCausalTransformer
from config import TRAIN_CFG as CFG, WANDB_PROJECT
from seed_utils import set_global_seed

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model: TinyCausalTransformer, optim: torch.optim.AdamW, ):
    pbar = tqdm(range(CFG["steps"]))
    for step in pbar:   
        #1. Generate Batch
        seq = generate_batch(
            batch_size=CFG['batch_size'],
            seq_len=CFG['seq_len'],
            vocab_size=CFG['vocab_size'],
            pattern_len=CFG['pattern_len'],
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
    
    torch.save(model.state_dict(), "tiny_transformer.pt")

if __name__ == "__main__":
    set_global_seed(CFG["seed"])
    CFG['run_name'] = "Induction_exp_0"

    wandb.init(
        project=WANDB_PROJECT,
        name=CFG['run_name'],
        config=CFG,
    )

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

    train(model=model, optim=optim)

    wandb.finish()
