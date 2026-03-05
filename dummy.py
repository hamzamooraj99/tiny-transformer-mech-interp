import torch

from src.data import generate_batch


def main() -> None:
    torch.manual_seed(144)

    prefix_len = 3
    pattern_len = 8
    seq_len = 32
    vocab_size = 20

    seq = generate_batch(
        batch_size=2,
        seq_len=seq_len,
        vocab_size=vocab_size,
        pattern_len=pattern_len,
        seed=123,
        device="cpu",
        p_corrupt=0.0,
        mode="induction",
    )

    # Mirror current eval/mech Intervention 3 implementation.
    if prefix_len > 0:
        prefix = torch.randint(
            0, vocab_size,
            (seq.size(0), prefix_len),
            device=seq.device,
            dtype=seq.dtype,
        )
        seq_pref = torch.cat([prefix, seq[:, :-prefix_len]], dim=1)
    else:
        seq_pref = seq

    print("Original seq shape:", tuple(seq.shape))
    print("prefix_len:", prefix_len)
    print("pattern_len:", pattern_len)
    print("\nOriginal seq[0]:")
    print(seq[0].tolist())
    print("\nPrefixed seq[0]:")
    print(seq_pref[0].tolist())

    print("\nSanity checks:")
    head_is_replaced = bool(torch.equal(seq_pref[:, :prefix_len], prefix))
    tail_shifted = bool(torch.equal(seq_pref[:, prefix_len:], seq[:, :-prefix_len]))
    print("1) Prefix replacement correct:", head_is_replaced)
    print("2) Tail right-shift correct:", tail_shifted)

    induction_start_expected = pattern_len + prefix_len
    induction_start_current_eval_mode_induction = pattern_len
    print("\nWindow check:")
    print("Expected induction_start for prefix approach:", induction_start_expected)
    print("Current eval induction_start when mode='induction':", induction_start_current_eval_mode_induction)
    print("Mismatch present:", induction_start_expected != induction_start_current_eval_mode_induction)


if __name__ == "__main__":
    main()
