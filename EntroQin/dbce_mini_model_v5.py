"""
dbce_mini_model_v5.py – DBC-e MiniModel v5 (Attention Hint) — Feb 14, 2026
==========================================================================
Low-RAM strong + lightweight self-attention hint for context awareness.
Defaults to Directed Birth Channel (DBC-e).
Operator Mode always on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DBCeOperator:
    @staticmethod
    def activate():
        print("\n" + "="*72)
        print("DBC-e OPERATOR MODE — HARMONY FIRST")
        print("4-Step Loop — Apply ruthlessly:")
        print("  D — Decide fast (hell-yes + 1–2 strongest cues)")
        print("  B — Build simplest viable path")
        print("  C — Check reality honestly (especially failure modes)")
        print("  e — Evolve or kill → tight loop")
        print("="*72 + "\n")

class DBCeCore(nn.Module):
    """v5 core with attention hint for cross-sequence context."""
    def __init__(self, dim=128, birth_triple=(0.01088, 0.06778, 0.00396), dropout=0.1):
        super().__init__()
        self.dim = dim
        self.birth_triple = torch.tensor(birth_triple, dtype=torch.float32)
        self.operator = DBCeOperator()

        self.ratchet = nn.Parameter(torch.ones(dim) * 0.55)
        self.resonance = nn.Linear(dim, dim)
        self.dissipation = nn.Parameter(torch.tensor(0.09))
        self.cross_link = nn.Parameter(torch.ones(dim) * 0.28)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # Lightweight attention hint (single-head, low compute)
        self.q_proj = nn.Linear(dim, dim // 4)
        self.k_proj = nn.Linear(dim, dim // 4)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_out = nn.Linear(dim, dim)

    def attention_hint(self, x):
        """Simple self-attention hint — adds context without heavy cost."""
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = F.softmax(q @ k.transpose(-2, -1) / (self.dim ** 0.5), dim=-1)
        return self.attn_out(attn @ v)

    def forward(self, x, prev_output=None):
        self.operator.activate()

        entropy_proxy = torch.mean(torch.abs(x), dim=-1, keepdim=True)
        harmony_cue = torch.sigmoid(-entropy_proxy)

        ratcheted = x * (1.0 + F.relu(self.ratchet))
        resonant = self.resonance(ratcheted)

        # Attention hint — context awareness
        attn_hint = self.attention_hint(resonant)
        resonant = resonant + 0.15 * attn_hint  # light residual

        noise = torch.randn_like(resonant) * self.dissipation
        dissipated = resonant - noise * harmony_cue
        normalized = self.norm(dissipated)
        dropped = self.dropout(normalized)

        if prev_output is not None:
            cross = self.cross_link * (dropped + prev_output) / 2.0
            output = dropped * 0.72 + cross * 0.28
        else:
            output = dropped

        target = self.birth_triple[1]
        harmony_loss = F.mse_loss(output.mean(), torch.tensor(target))
        output = output * (1.0 - 0.06 * harmony_loss)

        return output


class DBCeMiniModel(nn.Module):
    """v5 — 6-layer stack with attention hint."""
    def __init__(self, vocab_size=512, dim=128, n_layers=6, birth_triple=(0.01088, 0.06778, 0.00396)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([DBC eCore(dim, birth_triple, dropout=0.1) for _ in range(n_layers)])
        self.output_head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        prev = None
        for layer in self.layers:
            x = layer(x, prev)
            prev = x
        logits = self.output_head(x)
        return logits

# Quick test (same as before)
if __name__ == "__main__":
    # ... (use same Alice chapter test code as v4) ...