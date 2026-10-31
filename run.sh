#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
#  Multilingual AI Language Translator — Linux/Mac Launcher
# ──────────────────────────────────────────────────────────────────────

set -e

# Colors
CYAN='\033[0;36m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; RESET='\033[0m'

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║       🌐  MULTILINGUAL AI LANGUAGE TRANSLATOR  🌐       ║"
echo "  ║   Speak . Translate . Connect — 49 Languages            ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# Move to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo -e "${YELLOW}[1/3] Python version:${RESET}"
python3 --version

echo -e "${YELLOW}[2/3] Checking Gradio...${RESET}"
python3 -c "import gradio; print(f'  Gradio {gradio.__version__}')" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements.txt --quiet
}

echo -e "${YELLOW}[3/3] Starting app...${RESET}"
echo ""
echo -e "${GREEN}  ► Open in browser: http://localhost:7860${RESET}"
echo ""

python3 app/gradio_app.py
