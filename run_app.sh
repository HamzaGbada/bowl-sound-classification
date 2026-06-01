#!/usr/bin/env bash
cd "$(dirname "$0")"
uv run streamlit run app/streamlit_app.py --server.port 8501