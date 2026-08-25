#!/usr/bin/env bash
# Diligence-bench SDPO on 8x H100 (4 vLLM rollout + 4 FSDP trainer).
#
# Usage:
#   bash scripts/run_diligencebench.sh <mode> [extra dotted overrides...]
#
# Diligence is multi-turn TIR: the model may call Parallel web_search (mode=fast)
# then write a memo, scored by a rubric judge. Tool-result tokens are masked out
# of the SDPO loss; agent tokens between searches are SOD-reweighted.
#
# Modes
#   answer_free      Teacher hint must NOT state figures / conclusions. Main arm.
#   answer_bearing   Hint may cite the missed rubric facts verbatim. Stronger
#                    teacher, closer to supervised distillation.
#   mixture          50/50 KL mix of the two teachers.
#   baseline         Zero-shot rubric-judge score on the 30 held-out tasks.
#   smoke            1 GPU, tiny model, 10 steps.
#
# Examples
#   bash scripts/run_diligencebench.sh answer_free
#   bash scripts/run_diligencebench.sh answer_bearing trainer.total_steps=200
#   bash scripts/run_diligencebench.sh baseline
#
# Needs: 8x H100, PARALLEL_API_KEY (web search), OPENROUTER_API_KEY (hint +
#        rubric judge), WANDB_API_KEY
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
  sed -n '2,26p' "$0"
  exit 0
fi
shift

TS="$(date +%Y%m%d-%H%M%S)"
RUN="diligence-${MODE}-${TS}"
LOG_ROOT="${LOG_DIR:-/log}"
mkdir -p "${LOG_ROOT}/${RUN}" || {
  echo "cannot write ${LOG_ROOT}; set LOG_DIR=./log" >&2
  LOG_ROOT="./log"
  mkdir -p "${LOG_ROOT}/${RUN}"
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_PROJECT="${WANDB_PROJECT:-sdpo-diligence}"

NPROC=4
EXTRA=(
  data.dataset=diligence
  data.n_heldout=30
  logging.output_dir=runs/sdpo-diligence
  logging.wandb_project="$WANDB_PROJECT"
)
case "$MODE" in
  answer_free)
    EXTRA+=(generator.hint.prompt=answer_free)
    ;;
  answer_bearing)
    EXTRA+=(generator.hint.prompt=answer_bearing)
    ;;
  mixture)
    EXTRA+=(generator.hint.prompt=mixture)
    ;;
  baseline)
    EXTRA+=(--baseline generator.hint.prompt=answer_free)
    NPROC=1
    ;;
  smoke)
    EXTRA+=(--smoke generator.hint.prompt=answer_free)
    NPROC=1
    ;;
  *)
    echo "unknown mode ${MODE@Q}. try: answer_free answer_bearing mixture baseline smoke" >&2
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
  2>&1 | tee "${LOG_ROOT}/${RUN}/console.log"
