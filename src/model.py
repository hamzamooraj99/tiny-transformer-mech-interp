import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """
    Layer Normalization applied independently to each token representation.

    Given an input tensor of shape (B, T, d_model), this module normalizes
    each token vector across its feature dimension (d_model) to have
    zero mean and unit variance, then applies a learned affine transform:

        y = gamma * x_hat + beta

    where:
        x_hat = (x - mean) / sqrt(var + eps)

    Normalization is performed across the last dimension only, meaning
    statistics are computed independently for every token in every
    sequence (no dependence on batch statistics).

    Args:
        d_model (int):
            Dimensionality of the token representations.
        eps (float):
            Small constant added to variance for numerical stability.

    Shape:
        Input:  (B, T, d_model)
        Output: (B, T, d_model)

    Notes:
        - gamma and beta are learnable parameters of shape (d_model,).
        - Unlike BatchNorm, LayerNorm does not depend on batch size.
        - Commonly used in transformer architectures for training stability.
    """
    def __init__(self, d_model: int, eps: float=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention (decoder-style).

    This module implements scaled dot-product self-attention with multiple heads,
    including a causal mask to prevent tokens from attending to future positions.
    It follows the formulation introduced in "Attention Is All You Need"
    and used in decoder-only transformer architectures (e.g., GPT-style models).

    Given an input tensor of shape (B, T, d_model), the computation proceeds as:

        1. Linear projections to obtain queries (Q), keys (K), and values (V).
        2. Reshape into multiple heads of size d_head = d_model / n_heads.
        3. Compute scaled attention scores:
               scores = Q K^T / sqrt(d_head)
        4. Apply a causal mask to prevent attending to future tokens.
        5. Apply softmax over key dimension to obtain attention weights.
        6. Compute weighted sum of values.
        7. Concatenate heads and apply a final output projection.

    Args:
        d_model (int):
            Dimensionality of token representations.
        n_heads (int):
            Number of parallel attention heads. Must divide d_model evenly.

    Shape:
        Input:
            x: (B, T, d_model)
        Output:
            out: (B, T, d_model)

        If return_attn=True:
            returns (out, attn), where:
                attn: (B, n_heads, T, T)

    Notes:
        - Attention is causal: position i cannot attend to positions j > i.
        - Attention weights are computed independently per head.
        - The final output projection mixes information across heads.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, return_attn: bool=False) -> torch.Tensor:
        B, T, _ = x.shape # B -> Batch Size, T -> Sequence Length, _ -> d_model

        #Each of Q, K, and V have shape: (B, T, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        #Reshape Q, K, and V to (B, n_heads, T, d_head)
        Q = Q.view(B, T, self.n_heads, self.d_head)
        Q = Q.transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head)
        K = K.transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head)
        V = V.transpose(1, 2)

        #Memory fix
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        #Self attention calculation - Three steps: (1) scores = QK^T / sqrt(d_head) - (2) causal mask - (3) weights = softmax(scores), output = weights @ V
        #Step 1
        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.d_head ** 0.5) #Shape of scores -> (B, H, T, T) | For each head, for each query position i, a score against every key position j

        #Step 2 - stop token i from attending to future tokens -> create a (T, T) mask by broadcasting (T, T) mask across (B, H)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf")) 

        #Step 3a - Softmax over the last dimension (over keys)
        attn = F.softmax(scores, dim=-1) #shape (B, H, T, T)
        #Step 3b - Apply weights to values - V is (B, H, T, d_head)
        out = torch.matmul(attn, V)

        #Merge heads back to (B, T, d_model) - transpose then reshape
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)

        if return_attn:
            return out, attn
        
        return out #shape => (B, T, d_model)

class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network used within a transformer block.

    This module applies the same two-layer fully connected network
    independently to each token representation:

        MLP(x) = W2(GELU(W1(x)))

    where:
        - W1 expands the feature dimension from d_model to hidden_dim
        - GELU is a non-linear activation function
        - W2 projects the representation back to d_model

    The transformation is applied independently at each sequence position,
    meaning there is no interaction between tokens inside this module
    (token mixing is handled by the attention mechanism).

    Args:
        d_model (int):
            Dimensionality of token representations.
        hidden_dim (int):
            Size of the intermediate hidden layer (typically 4 * d_model).

    Shape:
        Input:  (B, T, d_model)
        Output: (B, T, d_model)

    Notes:
        - Commonly referred to as the "feed-forward" or "FFN" layer.
        - Provides non-linear feature transformation after attention.
    """
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x

class TransformerBlock(nn.Module):
    """
    Single pre-norm Transformer block.

    This block consists of two sub-layers:

        1. Multi-Head Causal Self-Attention
        2. Position-wise Feed-Forward Network (MLP)

    Each sub-layer is preceded by Layer Normalization and followed
    by a residual (skip) connection:

        x = x + Attention(LN(x))
        x = x + MLP(LN(x))

    This pre-norm formulation improves training stability and
    gradient flow, particularly when stacking multiple blocks.

    Args:
        d_model (int):
            Dimensionality of token representations.
        n_heads (int):
            Number of attention heads.
        mlp_hidden_dim (int):
            Hidden dimension size for the feed-forward network.

    Shape:
        Input:  (B, T, d_model)
        Output: (B, T, d_model)

    Notes:
        - Attention mixes information across tokens.
        - MLP transforms features independently per token.
        - Residual connections allow deep stacking.
    """
    def __init__(self, d_model: int, n_heads: int, mlp_hidden_dim: int):
        super().__init__()
        self.ln1 = LayerNorm(d_model=d_model) #First normalisation layer => pre-norm transformer (normalise before applying attention)
        self.attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads) #Token mixing component => allows tokens to attend to previous tokens, retrieve info, build context
        self.ln2 = LayerNorm(d_model=d_model) #Second normalisation layer => separate LN becuase attn changes representation. Re-normalise before applying MLP
        self.mlp = MLP(d_model=d_model, hidden_dim=mlp_hidden_dim) #MLP => Position-wise non-linear transformation - Does not mix tokens, operates independently on each token vector, allows building of complex feature transformations
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x)) #Pre-normalise [self.ln1(...)], Compute attention output [self.attn(...)], Add residual connection [x + ...] => Residual path for attention (learns a correction to x)
        x = x + self.mlp(self.ln2(x)) #Re-normalise [self.ln2(...)], Apply MLP position-wise transformation [self.mlp(...)], Add residual connection [x + ...] => Residual path for Feed-forward
        return x

class TinyCausalTransformer(nn.Module):
    """
    Tiny decoder-only (causal) transformer for next-token prediction.

    This model implements a minimal GPT-style architecture:
        - learned token embeddings
        - learned positional embeddings
        - a stack of pre-norm TransformerBlocks (causal self-attention + MLP)
        - final layer normalization
        - linear projection to vocabulary logits

    Given input token IDs of shape (B, T), the model outputs logits of shape
    (B, T, vocab_size), suitable for autoregressive next-token training with a
    causal attention mask inside each attention layer.

    Args:
        vocab_size (int):
            Number of discrete token IDs.
        d_model (int):
            Dimensionality of token representations.
        n_heads (int):
            Number of attention heads per block.
        n_layers (int):
            Number of stacked transformer blocks.
        mlp_hidden_dim (int):
            Hidden dimension of the MLP inside each block (typically 4 * d_model).
        max_seq_len (int):
            Maximum supported sequence length for positional embeddings.

    Shape:
        Input:
            idx: (B, T) integer token IDs in [0, vocab_size - 1]
        Output:
            logits: (B, T, vocab_size)

    Notes:
        - Causality is enforced via a triangular attention mask in the attention module.
        - The model does not apply softmax; probabilities are obtained by applying
        softmax to logits externally (e.g., inside the loss function).
    """
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, mlp_hidden_dim: int, max_seq_len: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model) #Maps token IDs into vectors - Input (B, T) --> Output (B, T, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model) #Learned positional embeddings - Gives each position 0..max_seq_len-1 a vector size of d_model

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, mlp_hidden_dim=mlp_hidden_dim) for _ in range(n_layers)]
        ) #List of Transformer Blocks, each with one Attention + MLP Unit.

        self.ln_f = LayerNorm(d_model=d_model) #Final normalisation (common GPT-style)
        self.head = nn.Linear(d_model, vocab_size, bias=False) #Projects final token representation to logits over vocabulary. Output is unnormalised scores; softmax applied later during loss computation

        self.max_seq_len = max_seq_len
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        #Get Shapes
        B, T = idx.shape
        assert T <= self.max_seq_len

        #Token Embeddings - Convert token IDs to token vectors
        x = self.tok_emb(idx)

        #Position Indicies - Creates positional information 
        pos = torch.arange(T, device=idx.device)
        pos = pos.unsqueeze(0)
        pos_emb = self.pos_emb(pos)

        #Add Position - Adds positional information to token vectors
        x = x + pos_emb

        #Blocks - Sequentially apply each transformer block
        for block in self.blocks:
            x = block(x)
        
        #Final Norm - Stability and cleaner logits
        x = self.ln_f(x)

        #Logits - Project to vocabulary logit
        logits = self.head(x)

        return logits



if __name__ == "__main__":
    # man = MultiHeadSelfAttention(d_model=8, n_heads=2)
    # random = torch.randn(2, 4, 8)
    # y, attn = man.forward(random, return_attn=True)
    # print(y.shape)
    # print(attn.shape)

    # mlp = MLP(d_model=8, hidden_dim=32)
    # x = torch.randn(2, 4, 8)
    # x = mlp.forward(x)
    # print(x.shape)

    # block = TransformerBlock(d_model=8, n_heads=2, mlp_hidden_dim=32)
    # x = torch.randn(2, 4, 8)
    # x = block.forward(x)
    # print(x.shape)

    model = TinyCausalTransformer(20, 8, 2, 2, 32, 32)
    idx = torch.randint(0, 20, (2, 4))
    logits = model.forward(idx=idx)
    print(logits.shape)