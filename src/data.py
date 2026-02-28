'''
# data.py
## Author: @hamzamooraj99 (Hamza Hassan Mooraj)
Description: Generates synthetic token sequences for an induction task
'''
import torch

def generate_induction_batch(batch_size: int, seq_len: int, vocab_size: int, pattern_len: int, seed: int|None=None, device: str="cpu", p_corrupt: float=0.0) -> torch.Tensor:
    """
    Generate a batch of synthetic token sequences for an induction-style task.

    Each sequence is constructed by sampling a random base pattern of length
    `pattern_len` and repeating it to fill `seq_len`. The resulting tensor
    has shape (batch_size, seq_len) and contains integer token IDs in the
    range [0, vocab_size - 1].

    Optionally, tokens in the second half of each sequence can be randomly
    corrupted with probability `p_corrupt`, enabling robustness and
    perturbation experiments.

    Args:
        batch_size (int):
            Number of independent sequences to generate.
        seq_len (int):
            Total length of each generated sequence.
        vocab_size (int):
            Number of distinct token IDs (0 to vocab_size - 1).
        pattern_len (int):
            Length of the base repeating pattern.
        seed (int | None):
            Optional random seed for deterministic generation.
        device (str):
            Device on which tensors are allocated ("cpu" or "cuda").
        p_corrupt (float):
            Probability of corrupting tokens in the second half of the sequence.

    Returns:
        torch.Tensor:
            LongTensor of shape (batch_size, seq_len) containing token IDs.
    """
    assert batch_size > 0
    assert seq_len > 1
    assert vocab_size > 1
    assert 1 <= pattern_len <= seq_len
    assert 0.0 <= p_corrupt <= 1.0

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    
    pattern = torch.randint(low=0, high=vocab_size, size=(batch_size, pattern_len), device=device, generator=gen)
    repeats = (seq_len + pattern_len - 1) // pattern_len
    seq = pattern.repeat(1, repeats)[:, :seq_len]

    uniform_random = torch.rand(batch_size, seq_len, device=device, generator=gen)
    mask = uniform_random < p_corrupt
    mask[:, :seq_len//2] = False
    # print(f"DTYPE: {mask.dtype}\nSHAPE: {mask.shape}\nCHECK: {mask[0]}")

    rand_tokens = torch.randint(low=0, high=vocab_size, size=seq.shape, device=device, generator=gen)
    # print(f"OG: {seq[0]}")
    seq = torch.where(mask, rand_tokens, seq)
    return seq

def generate_batch(batch_size: int, seq_len: int, vocab_size: int, pattern_len: int, seed: int|None=None, device: str="cpu", p_corrupt: float=0.0, mode: str="induction"):
    """
    Dispatch function for synthetic sequence generation.

    Currently supports:
        - "induction": Repeating-pattern sequences for induction experiments.

    Args:
        batch_size (int):
            Number of sequences to generate.
        seq_len (int):
            Length of each sequence.
        vocab_size (int):
            Number of distinct token IDs.
        pattern_len (int):
            Length of repeating pattern (for induction mode).
        device (str):
            Device for tensor allocation.
        p_corrupt (float):
            Corruption probability (mode-dependent).
        mode (str):
            Task type. Currently supports "induction".

    Returns:
        torch.Tensor:
            Batch of token sequences of shape (batch_size, seq_len).
    """
    if(mode == "induction"):
        return generate_induction_batch(batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size, pattern_len=pattern_len, seed=seed, device=device, p_corrupt=p_corrupt)
    elif(mode == "arithmetic"):
        raise ValueError("Arithmetic mode not implemented yet... Coming Soon")
    else:
        raise ValueError("Invalid mode: Select from \"induction\" or \"arithmetic\"")


# if __name__ == "__main__":
#     seq = generate_induction_instance(batch_size=4, seq_len=32, vocab_size=20, pattern_len=8, p_corrupt=0.5)
#     print(seq.shape)
#     print(f"NG: {seq[0]}")