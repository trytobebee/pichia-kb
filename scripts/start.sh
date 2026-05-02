#!/usr/bin/env bash
# Starts the kb-core web UI. Usage:  ./scripts/start.sh
set -e

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "❌ .env not found. Copy .env.example → .env and add your GEMINI_API_KEY first."
  exit 1
fi

if [ ! -d .venv ]; then
  echo "📦 Creating venv + installing dependencies (uv sync)..."
  uv sync
fi

# shellcheck disable=SC1091
source .venv/bin/activate
set -a; . .env; set +a

echo "🚀 Launching web UI at http://localhost:8501"
exec streamlit run web/Home.py
