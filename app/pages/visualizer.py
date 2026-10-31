"""
📊 Audio Visualizer Page
=========================
Visualizes the audio waveform and Mel spectrogram of uploaded/recorded audio.
This demonstrates the Audio-Spectrogram Transformer (AST) preprocessing step.
"""

import io
import numpy as np
import streamlit as st


def render_visualizer_page(audio_array: np.ndarray = None, sample_rate: int = 16000):
    """
    Render waveform + mel spectrogram visualization.

    Args:
        audio_array: 1-D float32 numpy array of audio samples.
        sample_rate: Sample rate of the audio.
    """
    st.markdown("## 📊 Audio Visualizer")
    st.caption(
        "This panel shows the raw waveform and 80-channel Log-Mel Spectrogram — "
        "the exact input fed into the **Audio-Spectrogram Transformer (Whisper)**."
    )

    if audio_array is None or len(audio_array) == 0:
        st.info("Record or upload audio first to see the visualization.")
        return

    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import librosa
        import librosa.display

        duration = len(audio_array) / sample_rate

        fig = plt.figure(figsize=(14, 8), facecolor="#0E1117")
        gs = gridspec.GridSpec(2, 1, hspace=0.45, figure=fig)

        # ── Waveform ──────────────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor("#1A1D27")
        time_axis = np.linspace(0, duration, len(audio_array))
        ax1.plot(time_axis, audio_array, color="#4FC3F7", linewidth=0.6, alpha=0.85)
        ax1.fill_between(time_axis, audio_array, alpha=0.15, color="#4FC3F7")
        ax1.set_title("🎙️ Audio Waveform", color="white", fontsize=13, pad=10)
        ax1.set_xlabel("Time (seconds)", color="#8892A4")
        ax1.set_ylabel("Amplitude", color="#8892A4")
        ax1.tick_params(colors="#8892A4")
        ax1.spines[:].set_color("#2D3748")
        ax1.set_xlim([0, duration])
        ax1.set_ylim([-1.1, 1.1])
        ax1.axhline(0, color="#2D3748", linewidth=0.5, linestyle="--")

        # ── Mel Spectrogram ───────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor("#1A1D27")

        mel = librosa.feature.melspectrogram(
            y=audio_array,
            sr=sample_rate,
            n_mels=80,
            n_fft=400,
            hop_length=160,
            fmin=0.0,
            fmax=8000.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        img = librosa.display.specshow(
            log_mel,
            sr=sample_rate,
            hop_length=160,
            x_axis="time",
            y_axis="mel",
            ax=ax2,
            cmap="magma",
        )
        fig.colorbar(img, ax=ax2, format="%+2.0f dB", pad=0.02)
        ax2.set_title(
            "🔊 80-Channel Log-Mel Spectrogram (AST Input)", color="white", fontsize=13, pad=10
        )
        ax2.set_xlabel("Time (seconds)", color="#8892A4")
        ax2.set_ylabel("Mel Frequency (Hz)", color="#8892A4")
        ax2.tick_params(colors="#8892A4")
        ax2.spines[:].set_color("#2D3748")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Stats ─────────────────────────────────────────────────────────────
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{duration:.2f}s")
        c2.metric("Sample Rate", f"{sample_rate:,} Hz")
        c3.metric("Mel Bins", "80")
        c4.metric("Samples", f"{len(audio_array):,}")

        # Architecture note
        st.info(
            "**How Whisper uses this:**  \n"
            "The 80-channel Log-Mel Spectrogram is split into **30-second patches** and "
            "fed through a **2-D convolutional stem** → flattened into tokens → "
            "passed through the **Transformer Encoder** (Multi-Head Attention + FFN). "
            "The encoder outputs are then cross-attended by the **auto-regressive Decoder** "
            "to generate transcribed text tokens."
        )

    except ImportError as e:
        st.error(f"Visualization requires matplotlib and librosa: {e}")
    except Exception as e:
        st.error(f"Visualization failed: {e}")
