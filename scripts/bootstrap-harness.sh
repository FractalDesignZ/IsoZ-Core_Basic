#!/usr/bin/env bash
# Linux bootstrap; run deliberately on a host with git, python3, npm and grok.
# Does not install system dependencies or write global user settings.
set -euo pipefail
REPO_URL="${SHEAF_REPO_URL:-https://github.com/FractalDesignZ/IsoZ-Core_Basic.git}"
INSTALL_DIR="${SHEAF_INSTALL_DIR:-${HOME}/isoz-core-basic}"
for tool in git python3 npm grok; do
  command -v "$tool" >/dev/null 2>&1 || { echo "Missing prerequisite: $tool" >&2; exit 1; }
done
python3 -c 'import numpy, scipy' || { echo 'Install numpy and scipy in your Python environment before bootstrapping.' >&2; exit 1; }
if [[ -d "$INSTALL_DIR" ]]; then
  cd "$INSTALL_DIR"
  [[ -z "$(git status --porcelain)" ]] || { echo 'Checkout has local changes; refusing to update.' >&2; exit 1; }
  git fetch origin phase-0
  if git show-ref --verify --quiet refs/heads/phase-0; then
    git checkout phase-0
  else
    git checkout --track -b phase-0 origin/phase-0
  fi
  git merge --ff-only origin/phase-0
else
  git clone --branch phase-0 --single-branch "$REPO_URL" "$INSTALL_DIR"
  cd "$INSTALL_DIR"
fi
grok plugin install . --trust
# Project settings contain both stdin-JSON PreToolUse and PostToolUse adapters.
python3 - <<'PY'
import json
from pathlib import Path
settings = json.loads(Path('.claude/settings.json').read_text())
for hook in ('PreToolUse', 'PostToolUse'):
    assert settings['hooks'][hook], f'Missing {hook}'
PY
mkdir -p reports
REPORT="reports/$(date +%Y-%m-%d_%H%M)_vm-bootstrap-verification-task-yield.md"
{
  echo '# VM bootstrap verification'
  echo "Commit: $(git rev-parse HEAD)"
  echo 'Plugin install completed. Project hooks are configured; host hook invocation must be checked in an interactive session.'
  echo 'The following are simulation reviews, not homology, model-fidelity, or hardware evidence.'
  python3 scripts/sheaf_coderabbit.py skills/harness-task-yield/SKILL.md --format markdown
  python3 scripts/sheaf_coderabbit.py agents/harness-backlog-reviewer.md --format markdown
} > "$REPORT"
echo "Bootstrap completed; actual review results are in $REPORT"
echo "Launch: cd $INSTALL_DIR && grok --plugin-dir ."
