"""
Model 3: FastSpeech2-Style Transformer TTS  (~20 Million Parameters)
=====================================================================
Non-autoregressive Text-to-Speech model that generates mel spectrograms
in parallel (unlike auto-regressive models like Tacotron2).

Architecture:
    Phoneme/Text Tokens
        ↓
    Phoneme Embedding (256 dim)
        ↓
    4× FFT Encoder Blocks:
        Multi-Head Self-Attention
        Conv1D FFN (kernel=9, instead of linear — captures local prosody)
        ↓
    Variance Adaptor:
        ├─ Duration Predictor → Length Regulator (upsample to mel frames)
        ├─ Pitch Predictor → Pitch Embedding (added to hidden states)
        └─ Energy Predictor → Energy Embedding (added to hidden states)
        ↓
    4× FFT Decoder Blocks (same structure as encoder)
        ↓
    Linear(256 → 80)  →  Mel Spectrogram (80 bins)
        ↓
    Vocoder:
        Griffin-Lim (iterative phase reconstruction, no training needed)
        OR HiFi-GAN (neural vocoder, pre-trained weights)
        ↓
    Audio Waveform (22050 Hz)

Training Losses:
    L_total = L_mel (MSE) + L_dur (MSE on log-durations) +
              L_pitch (MSE) + L_energy (MSE)

References:
    Ren et al. (2022) — FastSpeech 2: Fast and High-Quality TTS
    Kong et al. (2020) — HiFi-GAN: Generative Adversarial Networks for Audio
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
class TTSConfig:
    """Hyperparameters for the FastSpeech2-style TTS model."""
    # Vocabulary
    vocab_size: int = 200     # phoneme vocabulary (IPA or text characters)
    pad_id: int = 0

    # Model dimensions
    d_model: int = 256
    n_heads: int = 4
    n_fft_blocks: int = 4    # FFT blocks in encoder AND decoder
    conv_ff_kernel: int = 9  # kernel for Conv1D FFN
    d_ff: int = 1024
    dropout: float = 0.1

    # Audio
    n_mels: int = 80
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024

    # Variance predictor
    var_conv_channels: int = 256
    var_conv_kernel: int = 3
    var_dropout: float = 0.5

    # Pitch / energy quantization bins
    pitch_bins: int = 256
    energy_bins: int = 256


# ══════════════════════════════════════════════════════════════════════════════
# FFT Block (Feed-Forward Transformer Block)
# ══════════════════════════════════════════════════════════════════════════════

class ConvFFN(nn.Module):
    """
    Conv1D Feed-Forward Network used in FFT blocks.

    Replaces the standard linear FFN with two 1D convolutions.
    The larger kernel (k=9) captures local prosodic patterns
    (stress, rhythm) that linear layers miss.

    Structure:
        Conv1d(d → d_ff, k=9, pad=4) → ReLU → Dropout
        Conv1d(d_ff → d, k=1) → Dropout
    """

    def __init__(self, d_model: int, d_ff: int,
                 kernel: int = 9, dropout: float = 0.1):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=kernel, padding=pad)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.drop  = nn.Dropout(dropout)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)                                # (B, d, T)
        x = self.drop(F.relu(self.conv1(x)))
        x = self.drop(self.conv2(x))
        x = x.transpose(1, 2)                                # (B, T, d)
        return residual + x


class FFTBlock(nn.Module):
    """
    Feed-Forward Transformer (FFT) Block — core of FastSpeech2.

    Structure:
        LayerNorm → Multi-Head Self-Attention → residual
        ConvFFN (with Conv1D kernel=9)

    Using Conv1D instead of Linear in the FFN captures local temporal
    dependencies in phoneme sequences — critical for natural prosody.
    """

    def __init__(self, cfg: TTSConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.drop   = nn.Dropout(cfg.dropout)
        self.conv_ffn = ConvFFN(cfg.d_model, cfg.d_ff, cfg.conv_ff_kernel, cfg.dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:                (B, T, d_model)
            key_padding_mask: (B, T) — True for padding positions
        """
        # Pre-Norm self-attention
        n = self.norm(x)
        attn_out, _ = self.attn(n, n, n, key_padding_mask=key_padding_mask)
        x = x + self.drop(attn_out)
        # Conv1D FFN
        x = self.conv_ffn(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Variance Predictor (shared structure for Duration, Pitch, Energy)
# ══════════════════════════════════════════════════════════════════════════════

class VariancePredictor(nn.Module):
    """
    Shared structure for Duration, Pitch, and Energy predictors.

    Structure:
        Conv1d(d → 256, k=3, pad=1) → ReLU → LayerNorm → Dropout
        Conv1d(256 → 256, k=3, pad=1) → ReLU → LayerNorm → Dropout
        Linear(256 → 1)

    All three variance predictors share this architecture but have
    independent weights — they predict different acoustic properties.
    """

    def __init__(self, cfg: TTSConfig):
        super().__init__()
        c = cfg.var_conv_channels
        k = cfg.var_conv_kernel
        p = (k - 1) // 2
        d = cfg.dropout

        self.net = nn.Sequential(
            nn.Conv1d(cfg.d_model, c, kernel_size=k, padding=p),
            nn.ReLU(),
            nn.LayerNorm(c),                   # note: applied after transpose
            nn.Dropout(d),
            nn.Conv1d(c, c, kernel_size=k, padding=p),
            nn.ReLU(),
            nn.LayerNorm(c),
            nn.Dropout(d),
        )
        self.linear = nn.Linear(c, 1)
        self.d_model = cfg.d_model
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T)  predicted values (log-duration / pitch / energy)
        """
        B, T, D = x.shape
        out = x.transpose(1, 2)   # (B, D, T) for Conv1d
        # Apply conv layers manually (skip LayerNorm in sequential for shape)
        out = F.relu(self.net[0](out))    # Conv1
        out = out.transpose(1, 2)         # (B, T, c)
        out = self.net[2](out)            # LayerNorm
        out = self.net[3](out)            # Dropout
        out = out.transpose(1, 2)         # (B, c, T)
        out = F.relu(self.net[4](out))    # Conv2
        out = out.transpose(1, 2)
        out = self.net[6](out)
        out = self.net[7](out)
        return self.linear(out).squeeze(-1)   # (B, T)


# ══════════════════════════════════════════════════════════════════════════════
# Length Regulator
# ══════════════════════════════════════════════════════════════════════════════

class LengthRegulator(nn.Module):
    """
    Length Regulator — converts phoneme-level to frame-level representation.

    Given phoneme features and predicted durations (how many mel frames
    each phoneme spans), repeat each phoneme vector `duration` times.

    This is the key operation that makes FastSpeech2 non-autoregressive:
    the model predicts durations first, then generates all frames in parallel.

    Example:
        phonemes:  [P, AH, TH, ER]  durations: [2, 3, 2, 4]
        output:    [P, P, AH, AH, AH, TH, TH, ER, ER, ER, ER]
    """

    def forward(self, x: torch.Tensor,
                durations: torch.Tensor,
                max_len: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x:         (B, T_phoneme, d_model)
            durations: (B, T_phoneme)  integer durations per phoneme
            max_len:   optional maximum output length
        Returns:
            out:     (B, T_mel, d_model)
            mel_len: int
        """
        # Clamp durations to be non-negative integers
        durations = durations.round().long().clamp(min=0)
        B, T, D = x.shape

        output = []
        for b in range(B):
            frames = []
            for t in range(T):
                dur = durations[b, t].item()
                if dur > 0:
                    frames.append(x[b, t:t+1].expand(dur, -1))
            if frames:
                output.append(torch.cat(frames, dim=0))
            else:
                output.append(x.new_zeros(1, D))

        # Pad to same length
        mel_len = max(o.size(0) for o in output)
        if max_len is not None:
            mel_len = min(mel_len, max_len)

        padded = x.new_zeros(B, mel_len, D)
        for b, o in enumerate(output):
            l = min(o.size(0), mel_len)
            padded[b, :l] = o[:l]

        return padded, mel_len


# ══════════════════════════════════════════════════════════════════════════════
# Variance Adaptor
# ══════════════════════════════════════════════════════════════════════════════

class VarianceAdaptor(nn.Module):
    """
    Variance Adaptor — controls all acoustic variability.

    Steps:
    1. Predict log-durations → round to integers → Length Regulator
    2. Predict pitch per frame → quantize → add pitch embedding
    3. Predict energy per frame → quantize → add energy embedding

    During training: ground truth durations/pitch/energy used (teacher forcing).
    During inference: predicted values used.
    """

    def __init__(self, cfg: TTSConfig):
        super().__init__()
        self.dur_predictor    = VariancePredictor(cfg)
        self.pitch_predictor  = VariancePredictor(cfg)
        self.energy_predictor = VariancePredictor(cfg)
        self.length_regulator = LengthRegulator()

        # Quantize pitch/energy into bins and embed
        self.pitch_embedding  = nn.Embedding(cfg.pitch_bins + 1, cfg.d_model)
        self.energy_embedding = nn.Embedding(cfg.energy_bins + 1, cfg.d_model)

        self.pitch_bins  = cfg.pitch_bins
        self.energy_bins = cfg.energy_bins

    def _quantize(self, values: torch.Tensor, n_bins: int) -> torch.Tensor:
        """Quantize continuous values to bin indices (0 to n_bins)."""
        return (values.clamp(0, 1) * n_bins).long().clamp(0, n_bins)

    def forward(
        self,
        x: torch.Tensor,
        durations_gt: Optional[torch.Tensor] = None,
        pitch_gt: Optional[torch.Tensor] = None,
        energy_gt: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x:            (B, T_ph, d_model)  encoder output
            durations_gt: (B, T_ph)  ground truth durations (training only)
            pitch_gt:     (B, T_mel) ground truth pitch (training only)
            energy_gt:    (B, T_mel) ground truth energy (training only)
        Returns:
            out:      (B, T_mel, d_model)
            dur_pred: (B, T_ph)   log-duration predictions
            pitch_pred, energy_pred: (B, T_mel)
        """
        # 1. Duration prediction
        dur_pred = self.dur_predictor(x)              # (B, T_ph)  log-duration
        durations = durations_gt if durations_gt is not None else dur_pred.exp()

        # 2. Length regulation: expand phoneme → mel frames
        x, _ = self.length_regulator(x, durations)   # (B, T_mel, d)

        # 3. Pitch
        pitch_pred = self.pitch_predictor(x)          # (B, T_mel)
        pitch_vals = pitch_gt if pitch_gt is not None else pitch_pred
        pitch_emb  = self.pitch_embedding(self._quantize(pitch_vals, self.pitch_bins))
        x = x + pitch_emb

        # 4. Energy
        energy_pred = self.energy_predictor(x)        # (B, T_mel)
        energy_vals = energy_gt if energy_gt is not None else energy_pred
        energy_emb  = self.energy_embedding(self._quantize(energy_vals, self.energy_bins))
        x = x + energy_emb

        return x, dur_pred, pitch_pred, energy_pred


# ══════════════════════════════════════════════════════════════════════════════
# Griffin-Lim Vocoder (no training needed)
# ══════════════════════════════════════════════════════════════════════════════

class GriffinLimVocoder:
    """
    Griffin-Lim Algorithm for mel spectrogram → audio conversion.

    Iteratively estimates the phase of the STFT from magnitude (mel) spectrogram.
    No neural network — purely signal processing. Works out of the box.

    Quality: Good for demos and prototyping. For production, use HiFi-GAN.
    """

    def __init__(self, cfg: TTSConfig, n_iter: int = 32):
        self.cfg    = cfg
        self.n_iter = n_iter
        try:
            import librosa
            self.librosa = librosa
        except ImportError:
            self.librosa = None

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, n_mels, T) or (n_mels, T)  mel spectrogram
        Returns:
            waveform: (B, T_audio) or (T_audio,)
        """
        if self.librosa is None:
            # Return zeros if librosa not available
            T_audio = mel.shape[-1] * self.cfg.hop_length
            return torch.zeros(mel.shape[0] if mel.dim() == 3 else 1, T_audio)

        def _convert_one(mel_np):
            # Convert mel → linear spectrogram using librosa
            spec = self.librosa.feature.inverse.mel_to_stft(
                mel_np,
                sr=self.cfg.sample_rate,
                n_fft=self.cfg.n_fft,
            )
            return self.librosa.griffinlim(
                spec,
                n_iter=self.n_iter,
                hop_length=self.cfg.hop_length,
                win_length=self.cfg.win_length,
            )

        import numpy as np
        mel_np = mel.detach().cpu().numpy()
        if mel_np.ndim == 3:
            waves = [_convert_one(mel_np[b]) for b in range(mel_np.shape[0])]
            max_len = max(w.shape[0] for w in waves)
            padded = np.zeros((len(waves), max_len))
            for i, w in enumerate(waves):
                padded[i, :len(w)] = w
            return torch.from_numpy(padded).float()
        else:
            wave = _convert_one(mel_np)
            return torch.from_numpy(wave).float()


# ══════════════════════════════════════════════════════════════════════════════
# Full Model: FastSpeech2TTS
# ══════════════════════════════════════════════════════════════════════════════

class FastSpeech2TTS(nn.Module):
    """
    Model 3: FastSpeech2-Style Transformer TTS (~20M parameters)
    =============================================================
    Non-autoregressive TTS — generates entire mel spectrogram in one pass.

    Training:
        loss = MSE(mel_pred, mel_gt)
             + MSE(log_dur_pred, log_dur_gt)
             + MSE(pitch_pred, pitch_gt)
             + MSE(energy_pred, energy_gt)

    Inference:
        text_ids → mel_spectrogram → waveform (via Griffin-Lim)
    """

    def __init__(self, cfg: Optional[TTSConfig] = None):
        super().__init__()
        self.cfg = cfg or TTSConfig()
        cfg = self.cfg

        # Input embedding + positional encoding
        self.embedding  = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_drop   = nn.Dropout(cfg.dropout)

        # Encoder: 4 FFT blocks
        self.encoder    = nn.ModuleList([FFTBlock(cfg) for _ in range(cfg.n_fft_blocks)])
        self.enc_norm   = nn.LayerNorm(cfg.d_model)

        # Variance Adaptor
        self.var_adaptor = VarianceAdaptor(cfg)

        # Decoder: 4 FFT blocks
        self.decoder    = nn.ModuleList([FFTBlock(cfg) for _ in range(cfg.n_fft_blocks)])
        self.dec_norm   = nn.LayerNorm(cfg.d_model)

        # Mel projection head
        self.mel_linear = nn.Linear(cfg.d_model, cfg.n_mels)
        self.postnet    = PostNet(cfg)

        # Vocoder (no params — pure signal processing)
        self.vocoder    = GriffinLimVocoder(cfg)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _pos_encoding(self, T: int, device: torch.device) -> torch.Tensor:
        d   = self.cfg.d_model
        pe  = torch.zeros(T, d, device=device)
        pos = torch.arange(T, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2, device=device).float()
                        * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        """text_ids: (B, T_ph) → (B, T_ph, d_model)"""
        x = self.embedding(text_ids)
        x = self.pos_drop(x + self._pos_encoding(x.size(1), x.device))
        for block in self.encoder:
            x = block(x)
        return self.enc_norm(x)

    def forward(
        self,
        text_ids: torch.Tensor,
        durations_gt: Optional[torch.Tensor] = None,
        pitch_gt: Optional[torch.Tensor] = None,
        energy_gt: Optional[torch.Tensor] = None,
        mel_gt: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            text_ids:    (B, T_ph)   phoneme/character token ids
            durations_gt:(B, T_ph)   ground truth durations (training)
            pitch_gt:    (B, T_mel)  ground truth pitch (training)
            energy_gt:   (B, T_mel)  ground truth energy (training)
            mel_gt:      (B, n_mels, T_mel) ground truth mel (for loss)
        Returns:
            dict: mel_pred, mel_postnet, dur_pred, pitch_pred, energy_pred, loss
        """
        # 1. Encode phonemes
        enc_out = self.encode_text(text_ids)              # (B, T_ph, d)

        # 2. Variance adaptor
        x, dur_pred, pitch_pred, energy_pred = self.var_adaptor(
            enc_out, durations_gt, pitch_gt, energy_gt
        )                                                  # (B, T_mel, d)

        # 3. Add positional encoding to decoder input
        x = self.pos_drop(x + self._pos_encoding(x.size(1), x.device))

        # 4. Decode
        for block in self.decoder:
            x = block(x)
        x = self.dec_norm(x)                              # (B, T_mel, d)

        # 5. Mel projection
        mel_pred = self.mel_linear(x).transpose(1, 2)    # (B, n_mels, T_mel)
        mel_post = self.postnet(mel_pred) + mel_pred      # PostNet refinement

        result = {
            "mel_pred":    mel_pred,
            "mel_postnet": mel_post,
            "dur_pred":    dur_pred,
            "pitch_pred":  pitch_pred,
            "energy_pred": energy_pred,
            "loss":        None,
        }

        # 6. Compute losses (training mode)
        if mel_gt is not None:
            T_min = min(mel_pred.size(-1), mel_gt.size(-1))
            mel_loss  = F.mse_loss(mel_pred[..., :T_min], mel_gt[..., :T_min])
            post_loss = F.mse_loss(mel_post[..., :T_min], mel_gt[..., :T_min])

            dur_loss    = torch.tensor(0.0)
            pitch_loss  = torch.tensor(0.0)
            energy_loss = torch.tensor(0.0)

            if durations_gt is not None:
                log_dur_gt = torch.log(durations_gt.float().clamp(min=1))
                dur_loss = F.mse_loss(dur_pred, log_dur_gt)
            if pitch_gt is not None:
                T_p = min(pitch_pred.size(-1), pitch_gt.size(-1))
                pitch_loss = F.mse_loss(pitch_pred[..., :T_p], pitch_gt[..., :T_p])
            if energy_gt is not None:
                T_e = min(energy_pred.size(-1), energy_gt.size(-1))
                energy_loss = F.mse_loss(energy_pred[..., :T_e], energy_gt[..., :T_e])

            total_loss = mel_loss + post_loss + dur_loss + pitch_loss + energy_loss
            result["loss"]        = total_loss
            result["mel_loss"]    = mel_loss
            result["post_loss"]   = post_loss
            result["dur_loss"]    = dur_loss
            result["pitch_loss"]  = pitch_loss
            result["energy_loss"] = energy_loss

        return result

    @torch.no_grad()
    def synthesize(self, text_ids: torch.Tensor,
                   speed: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference: text_ids → mel spectrogram → waveform.
        Args:
            text_ids: (1, T_ph)  tokenized text
            speed:    float  — 1.0 = normal, <1 = faster, >1 = slower
        Returns:
            mel:   (1, n_mels, T_mel)
            audio: (1, T_audio)
        """
        self.eval()
        out = self.forward(text_ids)
        mel = out["mel_postnet"]                           # use refined mel

        # Apply speed control
        if speed != 1.0:
            T_new = int(mel.size(-1) / speed)
            mel = F.interpolate(mel, size=T_new, mode='linear', align_corners=False)

        audio = self.vocoder(mel)
        return mel, audio

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PostNet(nn.Module):
    """
    5-layer Conv1D PostNet for mel spectrogram refinement.
    Residual correction: adds fine details to the coarse mel prediction.

    Structure:
        4× [Conv1d(n_mels→256, k=5) → BatchNorm → Tanh → Dropout]
        1× [Conv1d(256→n_mels, k=5) → BatchNorm → Dropout]
    """

    def __init__(self, cfg: TTSConfig):
        super().__init__()
        layers = []
        in_ch  = cfg.n_mels
        for i in range(4):
            layers += [
                nn.Conv1d(in_ch, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.Tanh(),
                nn.Dropout(0.1),
            ]
            in_ch = 256
        layers += [
            nn.Conv1d(256, cfg.n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cfg.n_mels),
            nn.Dropout(0.1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (B, n_mels, T) → residual: (B, n_mels, T)"""
        return self.net(mel)


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

def demo():
    print("\n" + "=" * 60)
    print("  MODEL 3: FastSpeech2-Style TTS Demo")
    print("=" * 60)

    cfg = TTSConfig(
        vocab_size=50, d_model=64, n_heads=2, n_fft_blocks=2,
        d_ff=128, n_mels=80, conv_ff_kernel=3,
        var_conv_channels=64, var_conv_kernel=3,
        pitch_bins=32, energy_bins=32,
    )
    model = FastSpeech2TTS(cfg)

    B, T_ph, T_mel = 2, 15, 60

    text_ids    = torch.randint(1, 50, (B, T_ph))
    durations   = torch.ones(B, T_ph) * 4          # 4 frames per phoneme
    pitch_gt    = torch.rand(B, T_mel)
    energy_gt   = torch.rand(B, T_mel)
    mel_gt      = torch.randn(B, cfg.n_mels, T_mel)

    print(f"\nInput  text_ids : {text_ids.shape}")
    print(f"Input  mel_gt   : {mel_gt.shape}")

    enc = model.encode_text(text_ids)
    print(f"Encoder output  : {enc.shape}  (B, T_ph, d_model)")

    out = model(text_ids, durations, pitch_gt, energy_gt, mel_gt)
    print(f"Mel pred shape  : {out['mel_pred'].shape}")
    print(f"Mel postnet     : {out['mel_postnet'].shape}")
    print(f"Duration pred   : {out['dur_pred'].shape}")
    print(f"\nTraining losses:")
    print(f"  Total loss    : {out['loss'].item():.4f}")
    print(f"  Mel loss      : {out['mel_loss'].item():.4f}")
    print(f"  Duration loss : {out['dur_loss'].item():.4f}")

    mel, audio = model.synthesize(text_ids[:1])
    print(f"\nSynthesis mel   : {mel.shape}")
    print(f"Synthesis audio : {audio.shape}")

    params = model.count_parameters()
    print(f"\nTotal parameters: {params:,}  (~{params/1e6:.1f}M)")


if __name__ == "__main__":
    demo()
