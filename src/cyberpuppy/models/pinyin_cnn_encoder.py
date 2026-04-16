"""DBIT-style Depthwise CNN pinyin encoder for dual-branch architecture.

Adapted from: hjhhlc/DBIT (Expert Systems 2025, "Homophone-aware offensive
language detection via semantic-phonetic collaboration").

Architecture:
  pinyin chars → Embedding(vocab=35, dim=64) → DepthwiseConv1d → ReLU → PointwiseConv1d → AdaptiveMaxPool → (B, hidden_dim)

Total params: ~38K (vs Qwen3-8B's 7.6B). Negligible VRAM overhead.

The pinyin input is CHARACTER-LEVEL romanized text (e.g., "bai ren dou gai si"),
avoiding ALL BPE alignment issues. Homophones produce identical char sequences
by construction.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pinyin Character Tokenizer (char-level, deterministic, no BPE)
# ---------------------------------------------------------------------------

class PinyinCharTokenizer:
    """Maps pinyin string to integer IDs, one per character.

    Vocab: 26 lowercase letters + space + digits 0-9 + [PAD] + [UNK] = 39 tokens.
    """

    def __init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self.pad_id = 0
        self.unk_id = 1
        idx = 2
        # Space
        self._char_to_id[" "] = idx; idx += 1
        # Lowercase letters
        for c in "abcdefghijklmnopqrstuvwxyz":
            self._char_to_id[c] = idx; idx += 1
        # Digits
        for c in "0123456789":
            self._char_to_id[c] = idx; idx += 1
        self._vocab_size = idx

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> List[int]:
        """Encode pinyin string to char-level IDs."""
        return [self._char_to_id.get(c, self.unk_id) for c in text]

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 128,
    ) -> torch.Tensor:
        """Encode + pad/truncate a batch of pinyin strings.

        Returns: LongTensor (B, max_length), right-padded with pad_id.
        """
        batch = []
        for t in texts:
            ids = self.encode(t)[:max_length]
            pad = [self.pad_id] * (max_length - len(ids))
            batch.append(ids + pad)
        return torch.tensor(batch, dtype=torch.long)


# ---------------------------------------------------------------------------
# Depthwise Separable CNN Encoder (from DBIT Pinyin_Module.py)
# ---------------------------------------------------------------------------

class PinyinCNNEncoder(nn.Module):
    """Depthwise separable CNN for pinyin feature extraction.

    Exactly matches DBIT's PinyinEncoder: depthwise conv → ReLU pointwise conv → adaptive max pool.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim,
        )
        self.pointwise = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x: (B, seq_len, input_dim) — embedded pinyin characters

        Returns:
            (B, hidden_dim) — pooled pinyin representation
        """
        x = x.permute(0, 2, 1)         # (B, input_dim, seq_len)
        x = self.depthwise(x)
        x = F.relu(self.pointwise(x))  # (B, hidden_dim, seq_len)
        x = self.pool(x).squeeze(-1)   # (B, hidden_dim)
        return x


# ---------------------------------------------------------------------------
# PinyinBranch: Embedding + CNN (complete pinyin processing pipeline)
# ---------------------------------------------------------------------------

class PinyinBranch(nn.Module):
    """Complete pinyin branch: char embedding → CNN encoder → pooled vector.

    Takes integer pinyin char IDs (from PinyinCharTokenizer) and produces
    a fixed-size representation.
    """

    def __init__(
        self,
        vocab_size: int = 39,
        embed_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = PinyinCNNEncoder(embed_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, pinyin_ids: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            pinyin_ids: (B, L) integer char IDs from PinyinCharTokenizer

        Returns:
            (B, hidden_dim) pooled pinyin representation
        """
        x = self.embed(pinyin_ids)      # (B, L, embed_dim)
        return self.encoder(x)          # (B, hidden_dim)


# ---------------------------------------------------------------------------
# Contrastive Loss (bidirectional InfoNCE, exactly as DBIT Model.py)
# ---------------------------------------------------------------------------

def contrastive_loss(
    text_feat: torch.Tensor,
    pinyin_feat: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Bidirectional InfoNCE contrastive loss (DBIT style).

    Aligns text and pinyin representations: same-sample pairs are positives,
    cross-sample pairs are negatives. Homophones naturally get aligned because
    they share identical pinyin.

    Args:
        text_feat: (B, D) normalized text features
        pinyin_feat: (B, D) normalized pinyin features (same dim as text)
        temperature: softmax temperature (0.07 per DBIT default)

    Returns:
        Scalar loss
    """
    text_norm = F.normalize(text_feat, dim=1)
    pinyin_norm = F.normalize(pinyin_feat, dim=1)

    # Cosine similarity matrix (B, B)
    sim = torch.matmul(text_norm, pinyin_norm.T) / temperature

    labels = torch.arange(text_feat.size(0), device=text_feat.device)
    loss_t2p = F.cross_entropy(sim, labels)      # text → pinyin
    loss_p2t = F.cross_entropy(sim.T, labels)    # pinyin → text

    return (loss_t2p + loss_p2t) / 2
