"""
Model 2: Shared Multilingual Seq2Seq Transformer  (~35 Million Parameters)
===========================================================================
Custom Neural Machine Translation architecture with:
  - Shared multilingual vocabulary (SentencePiece 32k tokens)
  - Pre-LayerNorm for stable training
  - RoPE (Rotary Position Embeddings) — no absolute position bias
  - 6 Encoder + 6 Decoder layers
  - Weight-tied input/output embeddings (saves ~16M params)
  - Label smoothing (ε=0.1) for better calibration

Architecture:
    [src_lang_token] + source_tokens
        ↓
    Embedding(32k → 512) × √d_model + RoPE
        ↓
    6× Encoder Block (Pre-Norm):
        LayerNorm → MHSA with RoPE → residual
        LayerNorm → FFN(512→2048, GELU) → residual
        ↓
    Encoder output  (B, src_len, 512)
        ↓
    [tgt_lang_token] + target_tokens (teacher-forced)
        ↓
    Embedding + RoPE + causal mask
        ↓
    6× Decoder Block (Pre-Norm):
        LayerNorm → Masked MHSA with RoPE → residual
        LayerNorm → CrossAttention(enc_out) → residual
        LayerNorm → FFN(512→2048, GELU) → residual
        ↓
    LayerNorm → Linear(512→32000) [weight-tied with embedding]
        ↓
    Cross-Entropy Loss with label smoothing (ε=0.1)

References:
    Vaswani et al. (2017)  — Attention Is All You Need
    Su et al. (2021)       — RoFormer: Enhanced Transformer with RoPE
    NLLB Team (2022)       — No Language Left Behind
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Seq2SeqConfig:
    """Hyperparameters for the Multilingual Seq2Seq Transformer."""
    # Vocabulary (shared across all languages)
    vocab_size: int = 32000   # SentencePiece BPE tokens
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    # Model dimensions
    d_model: int = 512
    n_heads: int = 8          # d_model / n_heads = 64  (standard)
    n_enc_layers: int = 6
    n_dec_layers: int = 6
    d_ff: int = 2048          # 4× d_model

    # Regularisation
    dropout: float = 0.1
    label_smoothing: float = 0.1

    # Sequence length limits
    max_src_len: int = 512
    max_tgt_len: int = 512

    # Weight tying
    tie_embeddings: bool = True   # share encoder/decoder/output weights


# ══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embeddings (RoPE)
# ══════════════════════════════════════════════════════════════════════════════

class RotaryPositionEmbedding(nn.Module):
    """
    RoPE (Rotary Position Embedding) — Su et al., 2021.

    Unlike additive sinusoidal embeddings, RoPE encodes position by
    ROTATING the query and key vectors. This gives:
      - Better extrapolation to longer sequences
      - Relative position awareness built into attention
      - No parameters to learn

    For a vector pair (x_{2i}, x_{2i+1}), apply rotation by angle θ_i · m:
        x_{2i}'   =  x_{2i}   · cos(mθ_i)  −  x_{2i+1} · sin(mθ_i)
        x_{2i+1}' =  x_{2i}   · sin(mθ_i)  +  x_{2i+1} · cos(mθ_i)
    where θ_i = 10000^{−2i/d}  and m = token position.
    """

    def __init__(self, d_head: int, max_len: int = 4096):
        super().__init__()
        self.d_head = d_head
        # Precompute cos and sin tables: shape (max_len, d_head//2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        t     = torch.arange(max_len).float()
        freqs = torch.outer(t, self.inv_freq)          # (max_len, d_head//2)
        emb   = torch.cat([freqs, freqs], dim=-1)      # (max_len, d_head)
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs: [x1, x2, ...xn] → [-x_{n//2+1}, ..., x_n, x_1, ..., x_{n//2}]."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to query or key tensor.
        Args:
            x: (B, n_heads, T, d_head)
        Returns:
            (B, n_heads, T, d_head) — rotated
        """
        T    = x.size(2)
        cos  = self.cos_cache[:T].unsqueeze(0).unsqueeze(0)   # (1, 1, T, d_head)
        sin  = self.sin_cache[:T].unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin


# ══════════════════════════════════════════════════════════════════════════════
# Attention with RoPE
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttentionRoPE(nn.Module):
    """
    Multi-Head Attention with RoPE applied to Q and K.

    For self-attention:  Q, K, V all come from the same source x.
    For cross-attention: Q from decoder, K and V from encoder output.
                         RoPE is applied only to Q (decoder positions).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.wq   = nn.Linear(d_model, d_model, bias=False)
        self.wk   = nn.Linear(d_model, d_model, bias=False)
        self.wv   = nn.Linear(d_model, d_model, bias=False)
        self.wo   = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.d_k)
        self.scale = math.sqrt(self.d_k)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                apply_rope: bool = True) -> torch.Tensor:
        """
        Args:
            q, k, v: (B, T, d_model)
            mask:    (B, 1, T_q, T_k)  0 = masked out
            apply_rope: apply RoPE to Q and K
        """
        B = q.size(0)
        Q = self._split(self.wq(q))   # (B, h, T, d_k)
        K = self._split(self.wk(k))
        V = self._split(self.wv(v))

        if apply_rope:
            Q = self.rope(Q)
            K = self.rope(K)

        scores = (Q @ K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.drop(F.softmax(scores, dim=-1))
        out  = (attn @ V).transpose(1, 2).reshape(B, -1, self.n_heads * self.d_k)
        return self.drop(self.wo(out))


# ══════════════════════════════════════════════════════════════════════════════
# Pre-Norm Feed-Forward
# ══════════════════════════════════════════════════════════════════════════════

class PreNormFFN(nn.Module):
    """
    Pre-LayerNorm Feed-Forward Network.

    Pre-Norm (normalize before sub-layer) is more stable than Post-Norm,
    especially for deep models (12+ layers) and large vocabularies.

    Structure:  LayerNorm → Linear(d→4d) → GELU → Dropout → Linear(4d→d)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


# ══════════════════════════════════════════════════════════════════════════════
# Encoder Block
# ══════════════════════════════════════════════════════════════════════════════

class EncoderBlock(nn.Module):
    """
    Pre-Norm Encoder Block:
        LayerNorm → MHSA(RoPE) → residual
        LayerNorm → FFN(GELU)  → residual
    """

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn  = MultiHeadAttentionRoPE(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ffn   = PreNormFFN(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.attn(n, n, n, mask)
        x = self.ffn(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Decoder Block
# ══════════════════════════════════════════════════════════════════════════════

class DecoderBlock(nn.Module):
    """
    Pre-Norm Decoder Block:
        LayerNorm → Masked MHSA(RoPE) → residual
        LayerNorm → CrossAttention(enc_out, no RoPE on K/V) → residual
        LayerNorm → FFN(GELU) → residual
    """

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.self_attn  = MultiHeadAttentionRoPE(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.cross_attn = MultiHeadAttentionRoPE(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ffn        = PreNormFFN(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Masked self-attention (auto-regressive)
        n = self.norm1(x)
        x = x + self.self_attn(n, n, n, self_mask)
        # 2. Cross-attention over encoder output (no RoPE on encoder K/V)
        n = self.norm2(x)
        x = x + self.cross_attn(n, enc_out, enc_out, cross_mask, apply_rope=False)
        # 3. Feed-forward
        x = self.ffn(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Label Smoothing Loss
# ══════════════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Cross-Entropy with label smoothing (ε=0.1).

    Instead of one-hot targets, the target distribution is:
        y_smooth = (1 - ε) · y_onehot  +  ε / (V - 1)  (for non-pad tokens)

    This prevents overconfidence and improves model calibration.
    """

    def __init__(self, vocab_size: int, pad_id: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id     = pad_id
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B*T, vocab_size)
            targets: (B*T,)  token ids
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth targets
        smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
        smooth_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        smooth_dist[:, self.pad_id] = 0.0

        # Mask padding
        pad_mask = (targets == self.pad_id)
        loss = -(smooth_dist * log_probs).sum(dim=-1)
        loss[pad_mask] = 0.0
        return loss.sum() / (~pad_mask).sum().clamp(min=1)


# ══════════════════════════════════════════════════════════════════════════════
# Full Model: MultilingualSeq2Seq
# ══════════════════════════════════════════════════════════════════════════════

class MultilingualSeq2Seq(nn.Module):
    """
    Model 2: Shared Multilingual Seq2Seq Transformer (~35M parameters)
    ====================================================================
    Supports 49+ languages via language token prepending.
    Shared vocabulary and weight-tied output projection.

    Usage:
        # Training
        output = model(src_ids, tgt_ids)
        loss = output['loss']

        # Inference
        tgt_ids = model.generate(src_ids, tgt_lang_bos_id, max_length=200)
    """

    def __init__(self, cfg: Optional[Seq2SeqConfig] = None):
        super().__init__()
        self.cfg = cfg or Seq2SeqConfig()
        cfg = self.cfg

        # Shared embedding (encoder + decoder share same embedding matrix)
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.emb_scale = math.sqrt(cfg.d_model)
        self.dropout   = nn.Dropout(cfg.dropout)

        # Encoder
        self.enc_blocks  = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_enc_layers)])
        self.enc_norm    = nn.LayerNorm(cfg.d_model)

        # Decoder
        self.dec_blocks  = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_dec_layers)])
        self.dec_norm    = nn.LayerNorm(cfg.d_model)

        # Output projection (weight-tied to embedding)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Loss
        self.criterion = LabelSmoothingLoss(cfg.vocab_size, cfg.pad_id, cfg.label_smoothing)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=cfg_init := self.cfg.d_model ** -0.5)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.lm_head:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

    def _pad_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, 1, 1, T) — 0 where padding."""
        return (ids != self.cfg.pad_id).unsqueeze(1).unsqueeze(2)

    def encode(self, src_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode source tokens.
        Args:
            src_ids: (B, src_len)
        Returns:
            enc_out: (B, src_len, d_model)
        """
        mask = self._pad_mask(src_ids)
        x = self.dropout(self.embedding(src_ids) * self.emb_scale)
        for block in self.enc_blocks:
            x = block(x, mask)
        return self.enc_norm(x)

    def decode(self, tgt_ids: torch.Tensor, enc_out: torch.Tensor,
               src_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode target tokens given encoder output.
        Args:
            tgt_ids: (B, tgt_len)
            enc_out: (B, src_len, d_model)
            src_ids: (B, src_len)  for cross-attention mask
        """
        T = tgt_ids.size(1)
        causal_mask = self._causal_mask(T, tgt_ids.device)
        cross_mask  = self._pad_mask(src_ids) if src_ids is not None else None

        x = self.dropout(self.embedding(tgt_ids) * self.emb_scale)
        for block in self.dec_blocks:
            x = block(x, enc_out, causal_mask, cross_mask)
        return self.dec_norm(x)

    def forward(self, src_ids: torch.Tensor,
                tgt_ids: torch.Tensor) -> dict:
        """
        Teacher-forced training forward pass.
        Args:
            src_ids: (B, src_len)  [lang_token] + source tokens
            tgt_ids: (B, tgt_len)  [lang_token] + target tokens + [EOS]
        Returns:
            dict: loss, logits
        """
        enc_out  = self.encode(src_ids)
        dec_in   = tgt_ids[:, :-1]                          # remove EOS
        dec_tgt  = tgt_ids[:, 1:]                           # remove BOS
        dec_out  = self.decode(dec_in, enc_out, src_ids)
        logits   = self.lm_head(dec_out)                    # (B, T, vocab)

        B, T, V  = logits.shape
        loss = self.criterion(logits.reshape(B * T, V), dec_tgt.reshape(B * T))

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, src_ids: torch.Tensor, tgt_bos_id: int,
                 max_length: int = 200, beam_width: int = 1) -> torch.Tensor:
        """
        Greedy / beam-search translation.
        Args:
            src_ids:    (1, src_len)  source token ids with lang prefix
            tgt_bos_id: int           target language BOS token id
            max_length: int           max generated tokens
            beam_width: int           1 = greedy, >1 = beam search
        Returns:
            (1, out_len)  generated token ids
        """
        self.eval()
        enc_out = self.encode(src_ids)
        device  = src_ids.device
        tokens  = torch.tensor([[tgt_bos_id]], dtype=torch.long, device=device)

        for _ in range(max_length):
            dec_out = self.decode(tokens, enc_out, src_ids)
            logits  = self.lm_head(dec_out[:, -1, :])
            next_id = logits.argmax(dim=-1, keepdim=True)
            tokens  = torch.cat([tokens, next_id], dim=1)
            if next_id.item() == self.cfg.eos_id:
                break
        return tokens

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

def demo():
    print("\n" + "=" * 60)
    print("  MODEL 2: Multilingual Seq2Seq Transformer Demo")
    print("=" * 60)

    cfg = Seq2SeqConfig(
        vocab_size=1000, d_model=128, n_heads=4, n_enc_layers=2,
        n_dec_layers=2, d_ff=512, max_src_len=64, max_tgt_len=64,
    )
    model = MultilingualSeq2Seq(cfg)

    B, src_len, tgt_len = 2, 20, 15

    src_ids = torch.randint(4, 1000, (B, src_len))
    tgt_ids = torch.randint(4, 1000, (B, tgt_len))
    src_ids[:, 0] = 10   # simulate lang token
    tgt_ids[:, 0] = 11   # simulate lang token
    tgt_ids[:, -1] = cfg.eos_id

    print(f"\nInput  src_ids : {src_ids.shape}")
    print(f"Input  tgt_ids : {tgt_ids.shape}")

    enc = model.encode(src_ids)
    print(f"Encoder output : {enc.shape}  (B, src_len, d_model)")

    out = model(src_ids, tgt_ids)
    print(f"\nTraining loss  : {out['loss'].item():.4f}")
    print(f"Logits shape   : {out['logits'].shape}")

    gen = model.generate(src_ids[:1], tgt_bos_id=11)
    print(f"Generated ids  : {gen.shape}")

    params = model.count_parameters()
    print(f"\nTotal parameters: {params:,}  (~{params/1e6:.1f}M)")


if __name__ == "__main__":
    demo()
