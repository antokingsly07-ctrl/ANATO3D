#!/usr/bin/env bash
# ANATO3D — start the backend and frontend for local development (macOS/Linux).
#
#   bash scripts/run_dev.sh
#
# Backend  -> http://127.0.0.1:8000
# Frontend -> http://127.0.0.1:5173  (Vite dev server, proxies /api to backend)

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend="$root/backend"
frontend="$root/frontend"

command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 || {
    echo "Python 3.10+ is required but was not found on PATH." >&2; exit 1;
}
command -v npm >/dev/null 2>&1 || { echo "Node.js 18+ (npm) is required but was not found on PATH." >&2; exit 1; }

PY="$(command -v python3 || command -v python)"

cleanup() {
    echo
    echo "Stopping ANATO3D ..."
    [[ -n "${backend_pid:-}" ]] && kill "$backend_pid" 2>/dev/null || true
    [[ -n "${frontend_pid:-}" ]] && kill "$frontend_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting ANATO3D backend (http://127.0.0.1:8000) ..."
( cd "$backend" && "$PY" run.py ) &
backend_pid=$!

if [[ ! -d "$frontend/node_modules" ]]; then
    echo "Installing frontend dependencies (first run) ..."
    ( cd "$frontend" && npm install )
fi

echo "Starting ANATO3D frontend (http://127.0.0.1:5173) ..."
( cd "$frontend" && npm run dev ) &
frontend_pid=$!

echo
echo "Backend : http://127.0.0.1:8000/docs"
echo "Frontend: http://127.0.0.1:5173"
echo "Press Ctrl+C to stop both."

wait
