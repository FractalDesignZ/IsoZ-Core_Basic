#!/usr/bin/env bash
# Works with stdin hook JSON or the legacy target/content positional arguments.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v python3 >/dev/null 2>&1; then PYTHON=python3
elif command -v python >/dev/null 2>&1; then PYTHON=python
else
  echo "Sheaf hook requires Python 3; install the documented prerequisite." >&2
  exit 2
fi
exec "$PYTHON" "$ROOT/scripts/tool_use_hook.py" "$@"
