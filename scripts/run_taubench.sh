#!/usr/bin/env bash
# Tau-bench SDPO on 8 GPUs (4 vLLM rollout + 4 FSDP trainer).
#
# Usage:
#   bash scripts/run_taubench.sh <mode> [extra dotted overrides...]
#
# Modes
#   gold          Ablation A. Teacher sees Sierra gold docs (banking) / canonical
#                 tool trajectory (retail, airline). No hint LLM. Cheap. Default.
#   step_hint     Ablation B. After each rollout, OpenRouter DeepSeek names the
#                 SINGLE next correct action given the gold + transcript.
#   baseline      Zero-shot pass^1 on the official retail+airline test split.
#   smoke         1 GPU, tiny model, 10 steps. Sanity check the loop, not a result.
#   gold_banking  Same as gold but banking_knowledge only (97 tasks, no retail/airline).
#
# Examples
#   bash scripts/run_taubench.sh gold
#   bash scripts/run_taubench.sh step_hint trainer.total_steps=200
#   bash scripts/run_taubench.sh baseline
#
# Needs: 8 GPUs (4+4), OPENROUTER_API_KEY (tau2 user sim + DeepSeek hints), WANDB_API_KEY,
#        `uv sync --extra knowledge`, `bash scripts/setup_tau2_sandbox.sh`
#        `which` is not enough — bwrap must be able to create namespaces
#        (--privileged / seccomp=unconfined). run.py probes this and exits.
# Logs:  /log/<run_name>/{train.log,console.log,args.txt,config.yaml}
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$ROOT/.venv/bin:$ROOT/node_modules/.bin:$PATH"
if [[ -d "$ROOT/.deps/tau2-bench/data" ]]; then
  export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$ROOT/.deps/tau2-bench/data}"
fi
# vLLM 0.26 PyPI wheel needs libcudart.so.13; torch on this driver is cu128.
_cu13="$ROOT/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
if [[ -d "$_cu13" ]]; then
  export LD_LIBRARY_PATH="$_cu13${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

MODE="${1:-}"
if [[ -z "$MODE" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  sed -n '2,24p' "$0"
  exit 0
fi
shift

TS="$(date +%Y%m%d-%H%M%S)"
RUN="tau2-${MODE}-${TS}"
LOG_ROOT="${LOG_DIR:-/log}"
mkdir -p "${LOG_ROOT}/${RUN}" || {
  echo "cannot write ${LOG_ROOT}; set LOG_DIR=./log" >&2
  LOG_ROOT="./log"
  mkdir -p "${LOG_ROOT}/${RUN}"
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT="${WANDB_PROJECT:-sdpo-tau2}"

NPROC=4
EXTRA=()
case "$MODE" in
  gold)
    EXTRA+=(
      generator.hint.prompt=gold
      "data.domains=[retail,airline]"
      generator.engine.max_model_len=32768
    )
    ;;
  step_hint)
    EXTRA+=(generator.hint.prompt=step_hint)
    ;;
  gold_banking)
    EXTRA+=(
      generator.hint.prompt=gold
      "data.domains=[banking_knowledge]"
    )
    ;;
  baseline)
    EXTRA+=(
      --baseline
      generator.hint.prompt=gold
      "data.domains=[retail,airline]"
      generator.engine.max_model_len=32768
    )
    NPROC=1
    ;;
  smoke)
    EXTRA+=(--smoke generator.hint.prompt=gold)
    NPROC=1
    ;;
  *)
    echo "unknown mode ${MODE@Q}. try: gold step_hint baseline smoke gold_banking" >&2
    exit 1
    ;;
esac

if command -v uv >/dev/null 2>&1 && [[ -x "$ROOT/.venv/bin/python" || -f "$ROOT/pyproject.toml" ]]; then
  # --no-sync: the lockfile's torch is CUDA 13; this box is driver 12.8. A
  # sync would replace the cu128 stack (and vLLM) and break CUDA again.
  export UV_NO_SYNC=1
  PY=(uv run --no-sync python)
  TORCHRUN=(uv run --no-sync torchrun)
else
  PY=(python)
  TORCHRUN=(torchrun)
fi

if [[ "$NPROC" -gt 1 ]]; then
  LAUNCH=("${TORCHRUN[@]}" --standalone --nproc-per-node="$NPROC")
else
  LAUNCH=("${PY[@]}")
fi

echo "run ${RUN}"
echo "logs ${LOG_ROOT}/${RUN}"
echo "cmd  ${LAUNCH[*]} run.py ${EXTRA[*]} $* logging.run_name=${RUN} logging.log_dir=${LOG_ROOT}"

"${LAUNCH[@]}" run.py \
  "${EXTRA[@]}" \
  "$@" \
  logging.run_name="$RUN" \
  logging.log_dir="$LOG_ROOT" \
  logging.wandb_project="$WANDB_PROJECT" \
  2>&1 | tee "${LOG_ROOT}/${RUN}/console.log"
