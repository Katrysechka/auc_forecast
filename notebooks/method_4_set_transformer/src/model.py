"""Set Transformer with monotone-by-construction distributional reach head.

Architecture:
  per-user features -> MLP encoder -> N x ISAB -> PMA -> concat campaign feats
                                                              |
                                                              v
                                                       distributional head
                                                              |
                                                              v
                                  P(X = 0), P(X = 1), ..., P(X = k_max-1), P(X >= k_max)
                                                              |
                                                              v
                                  y_k = sum_{m >= k} P(X = m)  -- monotone by construction

Reference:
  Lee et al. 2019 — "Set Transformer: A Framework for Attention-based
                    Permutation-Invariant Neural Networks". (ISAB, PMA)

Ablation flags:
  use_attention=False  ->  mean-pool (Deep Sets) instead of ISAB+PMA
  use_distribution=False -> 3 sigmoid outputs (no monotone guarantee)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    """Multihead Attention Block — masked variant.

    Note on residual: when `d_q != d_out`, the input is projected by `fc_q` first and the
    residual is taken on the projected query. This is the original Set Transformer choice
    (Lee et al. 2019 §3) — keep it so reproduction matches the literature.
    """

    def __init__(self, d_q: int, d_k: int, d_out: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_out % n_heads == 0, f"d_out ({d_out}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.fc_q = nn.Linear(d_q, d_out)
        self.fc_k = nn.Linear(d_k, d_out)
        self.fc_v = nn.Linear(d_k, d_out)
        self.fc_o = nn.Linear(d_out, d_out)
        self.ln0 = nn.LayerNorm(d_out)
        self.ln1 = nn.LayerNorm(d_out)
        self.ff = nn.Sequential(
            nn.Linear(d_out, d_out * 2), nn.GELU(), nn.Linear(d_out * 2, d_out)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, mask_k: torch.Tensor | None = None) -> torch.Tensor:
        B, m, _ = Q.shape
        n = K.shape[1]
        q = self.fc_q(Q).view(B, m, self.n_heads, self.d_head).transpose(1, 2)
        k = self.fc_k(K).view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        v = self.fc_v(K).view(B, n, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask_k is not None:
            scores = scores.masked_fill(~mask_k[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, m, self.n_heads * self.d_head)
        out = self.fc_o(out)
        # Project Q via fc_q for residual (so dims match) — equivalent to taking the residual on `q` reshaped.
        q_skip = self.fc_q(Q) if Q.shape[-1] != out.shape[-1] else Q
        h = self.ln0(q_skip + out)
        h = self.ln1(h + self.ff(h))
        return h


class ISAB(nn.Module):
    """Induced Set Attention Block — O(n * m_ind) attention via inducing points."""

    def __init__(self, d_in: int, d_out: int, m_ind: int = 16, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, m_ind, d_out) * 0.02)
        self.mab1 = MAB(d_out, d_in, d_out, n_heads, dropout)
        self.mab2 = MAB(d_in, d_out, d_out, n_heads, dropout)

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B = X.shape[0]
        I = self.inducing.expand(B, -1, -1)
        H = self.mab1(I, X, mask_k=mask)          # [B, m_ind, d_out]
        out = self.mab2(X, H, mask_k=None)         # [B, n, d_out]
        return out


class PMA(nn.Module):
    """Pooling by Multihead Attention — k learnable seeds query the set."""

    def __init__(self, d: int, k: int = 1, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, k, d) * 0.02)
        self.mab = MAB(d, d, d, n_heads, dropout)

    def forward(self, X: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B = X.shape[0]
        S = self.seeds.expand(B, -1, -1)
        return self.mab(S, X, mask_k=mask)         # [B, k, d]


class SetTransformerReach(nn.Module):
    """End-to-end reach predictor.

    Variants (controlled at construction for clean ablations):
      use_attention=True   -> Set Transformer encoder (ISAB + PMA)
      use_attention=False  -> mean-pool over masked user MLP outputs (Deep Sets)
      use_distribution=True  -> distributional head with monotone tail-cumsum
      use_distribution=False -> 3 sigmoid outputs (NOT monotone — for ablation only)
    """

    def __init__(
        self,
        d_user: int,
        d_camp: int,
        n_publishers: int = 21,
        d_hidden: int = 96,
        n_isab: int = 2,
        m_ind: int = 16,
        n_heads: int = 4,
        k_max: int = 6,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_distribution: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_distribution = use_distribution
        self.k_max = k_max
        self.d_hidden = d_hidden

        self.user_enc = nn.Sequential(
            nn.Linear(d_user, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        if use_attention:
            self.isab = nn.ModuleList(
                [ISAB(d_hidden, d_hidden, m_ind, n_heads, dropout) for _ in range(n_isab)]
            )
            self.pma = PMA(d_hidden, k=1, n_heads=n_heads, dropout=dropout)
        else:
            self.isab = None
            self.pma = None

        self.camp_enc = nn.Sequential(
            nn.Linear(d_camp + n_publishers, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )

        head_in = d_hidden * 2
        head_out = (k_max + 1) if use_distribution else 3
        self.head = nn.Sequential(
            nn.Linear(head_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, head_out),
        )

    def forward(self, batch: dict) -> dict:
        X = batch["user_feats"]               # [B, n, D_user]
        mask = batch["mask"]                  # [B, n]
        camp = batch["camp_feats"]            # [B, D_camp]
        pub = batch["pub_mh"]                 # [B, n_pubs]

        H = self.user_enc(X)                  # [B, n, d_hidden]
        if self.use_attention:
            for blk in self.isab:
                H = blk(H, mask=mask)
            pooled = self.pma(H, mask=mask).squeeze(1)        # [B, d_hidden]
        else:
            m = mask.unsqueeze(-1).float()
            pooled = (H * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

        camp_repr = self.camp_enc(torch.cat([camp, pub], dim=1))
        joint = torch.cat([pooled, camp_repr], dim=1)
        logits = self.head(joint)

        if self.use_distribution:
            probs = F.softmax(logits, dim=1)                  # [B, k_max+1]
            # tail-cumsum: y_k = P(X >= k) = sum_{m=k..K} probs[m]
            cum_from_top = torch.cumsum(probs.flip(1), dim=1).flip(1)  # [B, k_max+1]
            y_hat = cum_from_top[:, 1:4]                       # [B, 3]
            return {"y_hat": y_hat, "probs": probs}
        else:
            y_hat = torch.sigmoid(logits)                      # ablation: no monotone guarantee
            return {"y_hat": y_hat, "probs": None}
