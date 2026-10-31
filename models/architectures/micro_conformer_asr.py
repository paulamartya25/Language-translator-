"""
Model 1: Micro-Conformer / Audio-Transformer ASR  (~25 Million Parameters)
===========================================================================
Custom architecture combining Audio-Spectrogram Transformer input processing
with a Conformer Encoder and Transformer Decoder.

Architecture Flow:
    Raw Audio (16 kHz mono)
        ↓
    Log-Mel Spectrogram  (80 bins, n_fft=400, hop=160)
        ↓
    2D Conv Subsampling  (reduces time dimension by 4×)
        Conv2d(1, 256, 3×3, stride=2, pad=1) → GELU → LayerNorm
        Conv2d(256, 256, 3×3, stride=2, pad=1) → GELU
        reshape → (B, T//4, 256)
        Linear(256 → d_model)
        ↓
    6× Conformer Encoder Block  (sandwich layout)
        FeedForward(½ residual)
        → RelativePositionMHSA
        → ConvolutionModule  (PW → GLU → DW-Conv k=31 → BN → Swish → PW)
        → FeedForward(½ residual)
        → LayerNorm
        ↓
    Encoder output  (B, T', d_model)
        ├── CTC head  Linear(d_model → vocab)  [CTC loss]
        └── 6× Transformer Decoder Block
                MaskedSelfAttention → CrossAttention(enc_out) → FFN
                ↓
            Linear(d_model → vocab)  [Cross-Entropy loss]

Joint loss = 0.7 × CE + 0.3 × CTC

References:
    Gulati et al. (2020) — Conformer: Conv-augmented Transformer for ASR
    Gong et al. (2021)  — AST: Audio Spectrogram Transformer
    Radford et al. (2022)— Robust Speech Recognition via Large-Scale Weak Supervision
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
class MicroConformerConfig:
    """Hyperparameters for the Micro-Conformer ASR model."""
    # Model dimensions
    d_model: int = 256        # main embedding / hidden dimension
    n_heads: int = 4          # attention heads  (d_model must be divisible by n_heads)
    n_enc_layers: int = 6     # Conformer encoder blocks
    n_dec_layers: int = 6     # Transformer decoder blocks
    d_ff: int = 1024          # feed-forward inner dimension
    conv_kernel_size: int = 31  # depthwise conv kernel in ConvolutionModule

    # Vocabulary
    vocab_size: int = 5000    # SentencePiece vocabulary size (including blank)
    blank_id: int = 0         # CTC blank token id
    bos_id: int = 1           # beginning-of-sequence token
    eos_id: int = 2           # end-of-sequence token
    pad_id: int = 3           # padding token

    # Audio front-end
    n_mels: int = 80          # log-mel spectrogram bins
    n_fft: int = 400          # FFT window size  (25 ms at 16 kHz)
    hop_length: int = 160     # STFT hop          (10 ms at 16 kHz)
    sample_rate: int = 16000

    # Regularisation
    dropout: float = 0.1

    # Training
    ctc_weight: float = 0.3   # weight of CTC loss in joint loss
    max_len: int = 3000       # max spectrogram frames


# ══════════════════════════════════════════════════════════════════════════════
# Audio Front-End: Log-Mel Spectrogram
# ══════════════════════════════════════════════════════════════════════════════

class LogMelSpectrogram(nn.Module):
    """
    Converts raw waveform → 80-channel log-Mel spectrogram.

    This is the input representation used by Audio-Spectrogram Transformers.
    The spectrogram is treated like a 2D image:  axes = (mel_bin, time_frame).

    Output shape: (batch, 1, n_mels, time_frames)
    """

    def __init__(self, cfg: MicroConformerConfig):
        super().__init__()
        self.cfg = cfg
        try:
            import torchaudio
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=cfg.sample_rate,
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                n_mels=cfg.n_mels,
                f_min=0.0,
                f_max=8000.0,
            )
        except ImportError:
            self.mel_transform = None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, time)  float32, values in [-1, 1]
        Returns:
            log_mel:  (batch, 1, n_mels, time_frames)
        """
        if self.mel_transform is not None:
            self.mel_transform = self.mel_transform.to(waveform.device)
            mel = self.mel_transform(waveform)              # (B, n_mels, T)
        else:
            # Fallback: simple STFT-based mel approximation
            mel = torch.stft(
                waveform.reshape(-1, waveform.shape[-1]),
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                return_complex=True,
            ).abs().pow(2)
            mel = mel[:, :self.cfg.n_mels, :]

        # Log compression (clamp to avoid log(0))
        log_mel = torch.log(mel.clamp(min=1e-9))           # (B, n_mels, T)
        return log_mel.unsqueeze(1)                         # (B, 1, n_mels, T)


# ══════════════════════════════════════════════════════════════════════════════
# 2D Conv Subsampling (Audio-Spectrogram Transformer style)
# ══════════════════════════════════════════════════════════════════════════════

class ConvSubsampling(nn.Module):
    """
    2D convolutional subsampling that reduces the time dimension by 4×.

    Architecture (identical to Whisper's conv stem):
        Conv2d(1 → d, 3×3, stride=2, pad=1) → GELU
        Conv2d(d → d, 3×3, stride=2, pad=1) → GELU
        reshape: (B, d, mel//4, T//4) → (B, T//4, d * mel//4)
        Linear(d * mel//4 → d_model)

    The 4× time reduction (2× per conv layer) makes attention
    computationally feasible for long audio sequences.
    """

    def __init__(self, cfg: MicroConformerConfig):
        super().__init__()
        d = cfg.d_model
        self.conv1 = nn.Conv2d(1, d, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d, d, kernel_size=3, stride=2, padding=1)
        self.act = nn.GELU()

        # After 2× downsampling on mel axis: n_mels // 4
        linear_in = d * (cfg.n_mels // 4)
        self.proj = nn.Linear(linear_in, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      (B, 1, n_mels, T)  — log-mel spectrogram
        Returns:
            out:    (B, T//4, d_model)
        """
        x = self.act(self.conv1(x))   # (B, d, n_mels//2, T//2)
        x = self.act(self.conv2(x))   # (B, d, n_mels//4, T//4)
        B, d, mel, t = x.shape
        x = x.permute(0, 3, 1, 2)    # (B, T//4, d, mel//4)
        x = x.reshape(B, t, d * mel) # (B, T//4, d * mel//4)
        x = self.norm(self.proj(x))  # (B, T//4, d_model)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Activation Functions
# ══════════════════════════════════════════════════════════════════════════════

class Swish(nn.Module):
    """Swish activation: f(x) = x · σ(x).  Smooth, non-monotonic, outperforms ReLU in ASR."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


# ══════════════════════════════════════════════════════════════════════════════
# Conformer Sub-Modules
# ══════════════════════════════════════════════════════════════════════════════

class FeedForwardModule(nn.Module):
    """
    Conformer Feed-Forward Module with ½ residual scaling.

    Structure:
        LayerNorm(x)
        → Linear(d → 4d) → Swish → Dropout
        → Linear(4d → d) → Dropout
        → ½ × residual add

    The ½ scaling is specific to Conformer (unlike standard Transformer).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1  = nn.Linear(d_model, d_ff)
        self.fc2  = nn.Linear(d_ff, d_model)
        self.act  = Swish()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return residual + 0.5 * x          # ½ residual scaling


class ConvolutionModule(nn.Module):
    """
    Conformer Convolution Module.

    Structure (all operations on time axis):
        LayerNorm
        → PointwiseConv(d → 2d, k=1)   → GLU gate  → d channels
        → DepthwiseConv(d, k=31, pad=15) [captures local phoneme patterns]
        → BatchNorm1d
        → Swish
        → PointwiseConv(d → d, k=1)
        → Dropout
        → residual add

    The depthwise conv with kernel=31 covers ~310ms of audio context
    at 10ms frame shift — ideal for capturing phonemes and syllables.
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        pad = (kernel_size - 1) // 2

        self.norm    = nn.LayerNorm(d_model)
        self.pw1     = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)  # → GLU
        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                 padding=pad, groups=d_model)
        self.bn      = nn.BatchNorm1d(d_model)
        self.pw2     = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act     = Swish()
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)               # (B, d, T)  for Conv1d
        x = self.pw1(x)                      # (B, 2d, T)
        x = F.glu(x, dim=1)                  # (B, d, T)  GLU gate
        x = self.act(self.bn(self.dw_conv(x)))
        x = self.drop(self.pw2(x))
        x = x.transpose(1, 2)               # (B, T, d)
        return residual + x


class RelativePositionMHSA(nn.Module):
    """
    Multi-Head Self-Attention with sinusoidal relative position bias.

    Relative positional encoding helps the model attend to nearby frames
    more strongly than distant ones — critical for speech where local
    context (phonemes, syllables) is most relevant.

    Attention score:  A[i,j] = (Q_i · K_j) / √d_k  +  r[i-j]
    where r[i-j] is a learned/sinusoidal relative position bias.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.norm = nn.LayerNorm(d_model)
        self.q    = nn.Linear(d_model, d_model)
        self.k    = nn.Linear(d_model, d_model)
        self.v    = nn.Linear(d_model, d_model)
        self.out  = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def _relative_pos_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Sinusoidal relative position bias: shape (1, 1, T, T)."""
        pos = torch.arange(seq_len, device=device)
        rel = pos.unsqueeze(0) - pos.unsqueeze(1)          # (T, T)
        # Encode with sin/cos
        bias = torch.zeros(seq_len, seq_len, device=device)
        for i, r in enumerate(range(-seq_len + 1, seq_len)):
            mask = (rel == r)
            if mask.any():
                bias[mask] = math.sin(r / (10000 ** (2 * (i % 64) / 128)))
        return bias.unsqueeze(0).unsqueeze(0)              # (1, 1, T, T)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    (B, T, d_model)
            mask: (B, 1, T, T)  optional padding mask
        """
        residual = x
        B, T, _ = x.shape
        x = self.norm(x)

        # Project → split heads
        def split(t):
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q = split(self.q(x))  # (B, h, T, d_k)
        K = split(self.k(x))
        V = split(self.v(x))

        # Scaled dot-product + relative position bias
        scores = (Q @ K.transpose(-2, -1)) / self.scale   # (B, h, T, T)
        scores = scores + self._relative_pos_bias(T, x.device)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.drop(F.softmax(scores, dim=-1))
        out  = (attn @ V).transpose(1, 2).reshape(B, T, self.d_model)
        return residual + self.drop(self.out(out))


class ConformerBlock(nn.Module):
    """
    One Conformer Encoder Block — the core of Model 1.

    Sandwich layout (Gulati et al., 2020):
        x → FF(½) → MHSA → ConvModule → FF(½) → LayerNorm → output

    This sandwich structure gives the model:
    - Local acoustic pattern capture (ConvolutionModule)
    - Global temporal context (MHSA)
    - Deep non-linear transformations (2× FFN)
    """

    def __init__(self, cfg: MicroConformerConfig):
        super().__init__()
        self.ff1  = FeedForwardModule(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.attn = RelativePositionMHSA(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.conv = ConvolutionModule(cfg.d_model, cfg.conv_kernel_size, cfg.dropout)
        self.ff2  = FeedForwardModule(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.ff1(x)
        x = self.attn(x, mask)
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════════════
# Conformer Encoder (6 blocks)
# ══════════════════════════════════════════════════════════════════════════════

class ConformerEncoder(nn.Module):
    """
    Stack of N Conformer Encoder blocks with sinusoidal position encoding.

    Processes the subsampled spectrogram features and produces rich
    contextual representations for both CTC decoding and cross-attention.
    """

    def __init__(self, cfg: MicroConformerConfig):
        super().__init__()
        self.pos_drop = nn.Dropout(cfg.dropout)
        self.blocks   = nn.ModuleList([ConformerBlock(cfg)
                                       for _ in range(cfg.n_enc_layers)])

    def _pos_encoding(self, T: int, d: int, device: torch.device) -> torch.Tensor:
        """Standard sinusoidal position encoding: shape (1, T, d)."""
        pe  = torch.zeros(T, d, device=device)
        pos = torch.arange(T, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2, device=device).float()
                        * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    (B, T, d_model)   subsampled spectrogram features
            mask: (B, 1, T, T)      optional padding mask
        Returns:
            (B, T, d_model)  encoder output
        """
        x = self.pos_drop(x + self._pos_encoding(x.size(1), x.size(2), x.device))
        for block in self.blocks:
            x = block(x, mask)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Transformer Decoder (6 blocks)
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Standard scaled dot-product Multi-Head Attention.
    Used for both masked self-attention (decoder) and cross-attention.

        Attention(Q, K, V) = softmax(QK^T / √d_k) · V
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.scale   = math.sqrt(self.d_k)

        self.q    = nn.Linear(d_model, d_model)
        self.k    = nn.Linear(d_model, d_model)
        self.v    = nn.Linear(d_model, d_model)
        self.out  = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, Tq, _ = q.shape

        def split(t):
            Bt, Tt, _ = t.shape
            return t.view(Bt, Tt, self.n_heads, self.d_k).transpose(1, 2)

        Q, K, V = split(self.q(q)), split(self.k(k)), split(self.v(v))
        scores = (Q @ K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.drop(F.softmax(scores, dim=-1))
        out  = (attn @ V).transpose(1, 2).reshape(B, Tq, -1)
        return self.drop(self.out(out))


class TransformerDecoderBlock(nn.Module):
    """
    One Transformer Decoder Block:
        1. Pre-Norm Masked Self-Attention   (auto-regressive)
        2. Pre-Norm Cross-Attention         (attends to encoder output)
        3. Pre-Norm Feed-Forward (GELU)
    """

    def __init__(self, cfg: MicroConformerConfig):
        super().__init__()
        d, h, ff, drop = cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)

        self.self_attn  = MultiHeadAttention(d, h, drop)
        self.cross_attn = MultiHeadAttention(d, h, drop)

        self.ff = nn.Sequential(
            nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop),
            nn.Linear(ff, d), nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Masked self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), self_mask)
        # 2. Cross-attention over encoder output
        x = x + self.cross_attn(self.norm2(x), enc_out, enc_out, cross_mask)
        # 3. Feed-forward
        x = x + self.ff(self.norm3(x))
        return x


class TransformerDecoder(nn.Module):
    """
    6-layer Transformer Decoder with:
    - Learned token embeddings
    - Sinusoidal position encoding
    - Causal (auto-regressive) mask generation
    """

    def __init__(self, cfg: MicroConformerConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_drop  = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([TransformerDecoderBlock(cfg)
                                        for _ in range(cfg.n_dec_layers)])
        self.norm      = nn.LayerNorm(cfg.d_model)
        self.d_model   = cfg.d_model

    def _pos_encoding(self, T: int, device: torch.device) -> torch.Tensor:
        d   = self.d_model
        pe  = torch.zeros(T, d, device=device)
        pos = torch.arange(T, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2, device=device).float()
                        * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask to prevent attending to future tokens."""
        return torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, tgt_ids: torch.Tensor,
                enc_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt_ids: (B, T_tgt)  decoder input token ids
            enc_out: (B, T_enc, d_model)  encoder output
        Returns:
            (B, T_tgt, d_model)
        """
        T = tgt_ids.size(1)
        x = self.embedding(tgt_ids) * math.sqrt(self.d_model)
        x = self.pos_drop(x + self._pos_encoding(T, tgt_ids.device))
        causal = self._causal_mask(T, tgt_ids.device)

        for block in self.blocks:
            x = block(x, enc_out, self_mask=causal)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════════════
# CTC Head
# ══════════════════════════════════════════════════════════════════════════════

class CTCHead(nn.Module):
    """
    CTC (Connectionist Temporal Classification) output head.
    Projects encoder output → vocabulary logits for CTC loss.
    CTC allows training without forced alignment between audio and text.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities for CTC: (T, B, vocab)."""
        return F.log_softmax(self.proj(self.norm(x)), dim=-1).transpose(0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Full Model: MicroConformerASR
# ══════════════════════════════════════════════════════════════════════════════

class MicroConformerASR(nn.Module):
    """
    Model 1: Micro-Conformer / Audio-Transformer ASR (~25M parameters)
    ====================================================================
    Complete end-to-end speech recognition model.

    Training:
        loss = 0.7 × CrossEntropyLoss(decoder) + 0.3 × CTCLoss(encoder)

    Inference:
        Greedy or beam-search decoding from decoder output.
    """

    def __init__(self, cfg: Optional[MicroConformerConfig] = None):
        super().__init__()
        self.cfg = cfg or MicroConformerConfig()
        cfg = self.cfg

        self.mel      = LogMelSpectrogram(cfg)
        self.subsamp  = ConvSubsampling(cfg)
        self.encoder  = ConformerEncoder(cfg)
        self.ctc_head = CTCHead(cfg.d_model, cfg.vocab_size)
        self.decoder  = TransformerDecoder(cfg)
        self.lm_head  = nn.Linear(cfg.d_model, cfg.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Audio → encoder output.
        Args:
            waveform: (B, T_audio)
        Returns:
            enc_out:  (B, T', d_model)
        """
        mel = self.mel(waveform)          # (B, 1, n_mels, T)
        x   = self.subsamp(mel)           # (B, T//4, d_model)
        return self.encoder(x)            # (B, T//4, d_model)

    def forward(
        self,
        waveform: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            waveform:       (B, T_audio)
            target_ids:     (B, T_tgt)   — shifted-right labels for CE loss
            target_lengths: (B,)         — for CTC loss
        Returns:
            Training mode (target_ids given):
                dict with keys: total_loss, ce_loss, ctc_loss, logits
            Inference mode:
                enc_out: (B, T', d_model)
        """
        enc_out = self.encode(waveform)

        # CTC path
        ctc_log_probs = self.ctc_head(enc_out)     # (T', B, vocab)

        if target_ids is None:
            return enc_out

        # Decoder path (teacher-forced)
        dec_in  = target_ids[:, :-1]               # remove EOS from input
        dec_tgt = target_ids[:, 1:]                # remove BOS from target
        dec_out = self.decoder(dec_in, enc_out)    # (B, T_tgt-1, d_model)
        logits  = self.lm_head(dec_out)            # (B, T_tgt-1, vocab)

        # Cross-entropy loss (decoder)
        B, T, V = logits.shape
        ce_loss = F.cross_entropy(
            logits.reshape(B * T, V),
            dec_tgt.reshape(B * T),
            ignore_index=self.cfg.pad_id,
        )

        # CTC loss (encoder)
        enc_lengths = torch.full((B,), enc_out.size(1), dtype=torch.long,
                                 device=waveform.device)
        if target_lengths is None:
            target_lengths = (dec_tgt != self.cfg.pad_id).sum(dim=1)

        ctc_loss = F.ctc_loss(
            ctc_log_probs,
            dec_tgt,
            enc_lengths,
            target_lengths,
            blank=self.cfg.blank_id,
            zero_infinity=True,
        )

        total = (1 - self.cfg.ctc_weight) * ce_loss + self.cfg.ctc_weight * ctc_loss

        return {
            "total_loss": total,
            "ce_loss":    ce_loss,
            "ctc_loss":   ctc_loss,
            "logits":     logits,
        }

    @torch.no_grad()
    def generate(self, waveform: torch.Tensor, max_length: int = 200) -> torch.Tensor:
        """
        Greedy decode: audio → token ids.
        Args:
            waveform: (1, T_audio)  single audio sample
        Returns:
            token_ids: (1, T_out)
        """
        self.eval()
        enc_out = self.encode(waveform)
        B = enc_out.size(0)
        device = waveform.device

        tokens = torch.full((B, 1), self.cfg.bos_id, dtype=torch.long, device=device)

        for _ in range(max_length):
            dec_out = self.decoder(tokens, enc_out)
            logits  = self.lm_head(dec_out[:, -1, :])
            next_id = logits.argmax(dim=-1, keepdim=True)
            tokens  = torch.cat([tokens, next_id], dim=1)
            if (next_id == self.cfg.eos_id).all():
                break
        return tokens

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

def demo():
    print("\n" + "=" * 60)
    print("  MODEL 1: Micro-Conformer ASR Demo")
    print("=" * 60)

    cfg = MicroConformerConfig(
        d_model=128, n_heads=2, n_enc_layers=2, n_dec_layers=2,
        d_ff=256, vocab_size=100, conv_kernel_size=7,
    )
    model = MicroConformerASR(cfg)

    B, T_audio = 2, 16000       # 1 second of audio, batch=2
    T_tgt = 20

    waveform = torch.randn(B, T_audio)
    tgt_ids  = torch.randint(4, 100, (B, T_tgt))
    tgt_ids[:, 0] = cfg.bos_id
    tgt_ids[:, -1] = cfg.eos_id

    print(f"\nInput  waveform : {waveform.shape}")

    mel   = model.mel(waveform)
    print(f"Log-Mel spec    : {mel.shape}  (B, 1, n_mels={cfg.n_mels}, T)")

    sub   = model.subsamp(mel)
    print(f"After subsampling: {sub.shape}  (B, T//4, d_model)")

    enc   = model.encoder(sub)
    print(f"Encoder output  : {enc.shape}  (B, T', d_model)")

    out   = model(waveform, tgt_ids)
    print(f"\nTraining losses:")
    print(f"  Total loss  : {out['total_loss'].item():.4f}")
    print(f"  CE loss     : {out['ce_loss'].item():.4f}")
    print(f"  CTC loss    : {out['ctc_loss'].item():.4f}")
    print(f"  Logits shape: {out['logits'].shape}")

    gen = model.generate(waveform[:1])
    print(f"\nGenerated tokens: {gen.shape}")

    params = model.count_parameters()
    print(f"\nTotal parameters: {params:,}  (~{params/1e6:.1f}M)")


if __name__ == "__main__":
    demo()
