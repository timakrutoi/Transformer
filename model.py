import torch
import torch.nn as nn
import torch.nn.functional as F


drop_p = 0.2


class AttentionHead(nn.Module):
    def __init__(self, head_size, block_size, n_emb):
        super().__init__()

        self.key = nn.Linear(n_emb, head_size)
        self.querry = nn.Linear(n_emb, head_size)
        self.value = nn.Linear(n_emb, head_size)

        self.dropout = nn.Dropout(drop_p)

        # special torch thing
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.querry(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        return wei @ v


class FeedForward(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(size, 4 * size),
            nn.ReLU(),
            nn.Linear(4 * size, size),
            nn.Dropout(drop_p)
        )
    
    def forward(self, x):
        return self.nn(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, block_size, n_emb):
        super().__init__()

        self.heads = nn.ModuleList([AttentionHead(head_size, block_size, n_emb) for _ in range(num_heads)])
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        r = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(r))


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, block_size, n_emb):
        super().__init__()

        head_size = n_emb // num_heads
        self.heads = MultiHeadAttention(num_heads=num_heads, head_size=head_size, block_size=block_size, n_emb=n_emb)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
    
    def forward(self, x):
        x = x + self.heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, block_size, n_emb, num_heads, num_transformer_blocks):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, n_emb)
        self.pos_emb = nn.Embedding(block_size, n_emb)

        self.blocks = nn.Sequential()

        for _ in range(num_transformer_blocks):
            self.blocks.append(TransformerBlock(num_heads, block_size, n_emb))
        
        self.blocks.append(nn.LayerNorm(n_emb))

        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, x):
        B,T = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T))

        x = tok + pos
        x = self.blocks(x)
        x = self.lm_head(x)

        return x
