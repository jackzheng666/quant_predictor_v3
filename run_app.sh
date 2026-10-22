#!/bin/sh
# Resolve the workspace even when launched from another directory.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$project_dir"
exec "$project_dir/.venv/bin/python" -m streamlit run "$project_dir/app.py" \
    --server.address 127.0.0.1 --server.port 8501 --server.headless true \
    --server.fileWatcherType poll --browser.gatherUsageStats false "$@"
