#!/usr/bin/env bash
# ANATO3D — run the full test suite (backend + frontend).
#
#   bash scripts/run_tests.sh

set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(command -v python3 || command -v python)"

echo "==> Backend tests (pytest)"
( cd "$root/backend" && "$PY" -m pytest -q )
backend_exit=$?

echo "==> Frontend typecheck"
( cd "$root/frontend" && npm run typecheck )
typecheck_exit=$?

echo "==> Frontend tests (vitest)"
( cd "$root/frontend" && npm test )
frontend_exit=$?

echo
if [[ $backend_exit -eq 0 && $typecheck_exit -eq 0 && $frontend_exit -eq 0 ]]; then
    echo "All checks passed."
    exit 0
fi

echo "One or more checks failed."
exit 1
