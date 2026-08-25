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
# And that `srt -c 'echo srt-ok'` actually prints srt-ok. `which` is not enough:
# many GPU pods install bwrap but block Linux namespaces via seccomp, so every
# shell tool call then fails with "Creating new namespace failed".
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ripgrep bubblewrap socat nodejs npm
else
  echo "not a Debian/Ubuntu host -- install ripgrep, bubblewrap (Linux) or sandbox-exec (macOS), socat, and node/npm yourself" >&2
fi

# Pin 0.0.23. 0.0.24+ has a Linux bwrap stub regression.
# Prefer a project-local install so we do not need a global npm prefix.
npm install --prefix "$ROOT" @anthropic-ai/sandbox-runtime@0.0.23
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  ln -sfn "$ROOT/node_modules/.bin/srt" "$ROOT/.venv/bin/srt"
fi
export PATH="$ROOT/.venv/bin:$ROOT/node_modules/.bin:$PATH"

echo "sandbox tools:"
command -v srt || echo "  srt: MISSING"
command -v rg || echo "  rg: MISSING"
command -v bwrap || echo "  bwrap: MISSING (ok on macOS -- uses sandbox-exec)"
command -v socat || echo "  socat: MISSING"

echo
echo "namespace smoke (required on Linux):"
if bwrap --ro-bind / / --dev /dev /bin/echo bwrap-ok >/dev/null 2>&1; then
  echo "  bwrap: ok"
else
  echo "  bwrap: FAILED -- this container blocks Linux namespaces (seccomp)." >&2
  echo "  Recreate the pod privileged, or with Docker:" >&2
  echo "    --privileged" >&2
  echo "    # or, narrower:" >&2
  echo "    --security-opt seccomp=unconfined --security-opt apparmor=unconfined" >&2
  echo "  On RunPod: Edit Pod -> Docker command / extra flags, add --privileged, then Start." >&2
  echo "  After reboot, re-run:  srt -c 'echo srt-ok'" >&2
  exit 1
fi

srt -c 'echo srt-ok'
echo "srt: ok"
