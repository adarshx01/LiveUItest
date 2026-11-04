#!/usr/bin/env bash
# Creates a project-local PLAYWRIGHT_BROWSERS_PATH whose revisions match the pinned
# Playwright MCP, reusing browser/ffmpeg builds that are already installed on the host.
# Use this only where `npx playwright install chromium ffmpeg` cannot download (air-gapped
# CI, restricted sandboxes). Container images install the exact revisions instead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="${HOST_BROWSERS_PATH:-/opt/pw-browsers}"
DEST="${1:-$ROOT/data/pw-browsers}"
BROWSERS_JSON="$ROOT/browser/playwright-mcp/node_modules/playwright-core/browsers.json"
mkdir -p "$DEST"
for name in chromium ffmpeg chromium_headless_shell; do
  key="$name"; [ "$name" = "chromium_headless_shell" ] && key="chromium-headless-shell"
  rev=$(python3 -c "import json,sys; print(next(b['revision'] for b in json.load(open('$BROWSERS_JSON'))['browsers'] if b['name']=='$key'))")
  have=$(ls -d "$SRC/${name}-"* 2>/dev/null | sort | tail -1 || true)
  if [ -n "$have" ] && [ ! -e "$DEST/${name}-${rev}" ]; then
    ln -s "$have" "$DEST/${name}-${rev}"
    echo "linked $DEST/${name}-${rev} -> $have"
  fi
done
echo "export AITEST_PLAYWRIGHT_BROWSERS_PATH=$DEST"
