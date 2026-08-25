#!/usr/bin/env bash
# Host binaries for tau2 banking_knowledge alltools-qwen.
#
# Retail and airline do not need this. banking_knowledge's `shell` tool runs
# inside Anthropic sandbox-runtime (`srt`), which shells out to rg/bwrap/socat
# and raises SandboxRuntimeError at env construction if any of them is missing.
#
#   uv sync --extra knowledge
#   bash scripts/setup_tau2_sandbox.sh
#
# Then verify:  which srt rg bwrap socat
set -euo pipefail

npm install -g @anthropic-ai/sandbox-runtime@0.0.23

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ripgrep bubblewrap socat
else
  echo "not a Debian/Ubuntu host -- install ripgrep, bubblewrap (Linux) or sandbox-exec (macOS), and socat yourself" >&2
fi

echo "sandbox tools:"
command -v srt || echo "  srt: MISSING"
command -v rg || echo "  rg: MISSING"
command -v bwrap || echo "  bwrap: MISSING (ok on macOS -- uses sandbox-exec)"
command -v socat || echo "  socat: MISSING"
