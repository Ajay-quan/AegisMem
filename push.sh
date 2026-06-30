#!/usr/bin/env bash
# One-shot: commit the v0.3.0 upgrade and push to GitHub.
# Run from the repo root:  bash push.sh
set -e

cd "$(dirname "$0")"

# Clear any stale git lock (safe: only removes an empty lock file).
rm -f .git/index.lock 2>/dev/null || true

echo "Staging changes…"
git add -A

echo "Committing…"
git commit -m "v0.3.0: Stateful-CL continual learning, telemetry, PII redaction, SDK/CLI, docs" || {
  echo "Nothing to commit (already committed?) — continuing to push."
}

echo "Pushing to origin main…"
git push origin main

echo "Done. Check https://github.com/Ajay-quan/stateful.ai"
