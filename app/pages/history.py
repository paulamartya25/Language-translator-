"""
📜 Translation History Page
============================
Shows all past translation sessions with search and replay capability.
"""

import json
import streamlit as st
from pathlib import Path
from datetime import datetime


HISTORY_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "history.json"


def save_to_history(result: dict):
    """Save a translation result to persistent history."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "transcription": result.get("transcription", ""),
        "source_language": result.get("source_language", ""),
        "translations": {
            lang: data.get("text", "")
            for lang, data in result.get("translations", {}).items()
        },
    }

    history = []
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            history = []

    history.insert(0, entry)
    history = history[:50]  # Keep last 50 entries

    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def load_history() -> list:
    """Load translation history from file."""
    if not HISTORY_FILE.exists():
        return []
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def render_history_page():
    """Render the translation history page."""
    st.markdown("## 📜 Translation History")
    st.caption("Last 50 translation sessions are saved automatically.")

    history = load_history()

    if not history:
        st.info("No translation history yet. Run a translation first!")
        return

    # Search
    search = st.text_input("🔍 Search in history", placeholder="Type to filter...")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear All History", type="secondary"):
            if HISTORY_FILE.exists():
                HISTORY_FILE.unlink()
            st.success("History cleared!")
            st.rerun()

    st.divider()

    for i, entry in enumerate(history):
        ts = entry.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts)
            ts_str = dt.strftime("%d %b %Y, %I:%M %p")
        except Exception:
            ts_str = ts

        transcription = entry.get("transcription", "")
        source_lang = entry.get("source_language", "Unknown")
        translations = entry.get("translations", {})

        # Filter by search
        if search and search.lower() not in transcription.lower():
            skip = True
            for t in translations.values():
                if search.lower() in t.lower():
                    skip = False
                    break
            if skip:
                continue

        with st.expander(
            f"**{ts_str}** — {source_lang}: *{transcription[:80]}{'...' if len(transcription)>80 else ''}*"
        ):
            st.markdown(f"**Original ({source_lang}):** {transcription}")
            st.markdown(f"**Languages translated:** {len(translations)}")

            if translations:
                lang_cols = st.columns(3)
                for j, (lang, text) in enumerate(translations.items()):
                    with lang_cols[j % 3]:
                        st.markdown(f"**{lang}**")
                        st.caption(text[:120] + ("..." if len(text) > 120 else ""))

            # Re-export
            st.download_button(
                label="⬇️ Download this session",
                data=json.dumps(entry, ensure_ascii=False, indent=2),
                file_name=f"translation_{i}.json",
                mime="application/json",
                key=f"dl_{i}",
            )
