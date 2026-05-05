#!/bin/bash
#SBATCH --job-name=mta_bus
#SBATCH --output=logs/mta_%j.out
#SBATCH --error=logs/mta_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# ── configuration ─────────────────────────────────────────────────────────────
# Required: set DATA_DIR to the folder containing your parsed CSVs.
#   sbatch --export=ALL,DATA_DIR=/path/to/parsed run_slurm.sh
#
# Optional overrides (pass via --export=ALL,...):
#   MODEL        : rnn | rf | both          (default: rnn)
#   HIDDEN       : LSTM hidden size         (default: 128)
#   LAYERS       : LSTM layers              (default: 2)
#   DROPOUT      : dropout rate             (default: 0.2)
#   LR           : learning rate            (default: 1e-3)
#   BATCH        : batch size               (default: 64)
#   EPOCHS       : max epochs               (default: 100)
#   PATIENCE     : early-stop patience      (default: 10)
#   RF_SEARCH    : RF random-search iters   (default: 20)
#   CACHE_DIR    : trip cache directory     (default: trips_cache)
#   OUT_DIR      : output directory         (default: outputs/<job-id>)

if [ -z "${DATA_DIR}" ]; then
    echo "ERROR: DATA_DIR is not set."
    echo "  sbatch --export=ALL,DATA_DIR=/path/to/parsed run_slurm.sh"
    exit 1
fi

# Run from the directory sbatch was called from so relative paths work.
cd "${SLURM_SUBMIT_DIR}"

MODEL="${MODEL:-rnn}"
HIDDEN="${HIDDEN:-128}"
LAYERS="${LAYERS:-2}"
DROPOUT="${DROPOUT:-0.2}"
LR="${LR:-1e-3}"
BATCH="${BATCH:-64}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-10}"
RF_SEARCH="${RF_SEARCH:-20}"
RF_MAX_SAMPLES="${RF_MAX_SAMPLES:-0.5}"
MIN_TRIP_LEN="${MIN_TRIP_LEN:-10}"
CACHE_DIR="${CACHE_DIR:-trips_cache}"
OUT_DIR="${OUT_DIR:-outputs/${SLURM_JOB_ID}}"

mkdir -p logs "${OUT_DIR}" "${CACHE_DIR}"

echo "=========================================="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "Started    : $(date)"
echo "GPU        : ${CUDA_VISIBLE_DEVICES:-<auto>}"
echo "DATA_DIR   : ${DATA_DIR}"
echo "MODEL      : ${MODEL}"
echo "HIDDEN     : ${HIDDEN}  LAYERS: ${LAYERS}  DROPOUT: ${DROPOUT}"
echo "LR         : ${LR}  BATCH: ${BATCH}"
echo "EPOCHS     : ${EPOCHS}  PATIENCE: ${PATIENCE}"
echo "CACHE_DIR  : ${CACHE_DIR}"
echo "OUT_DIR    : ${OUT_DIR}"
echo "=========================================="

module load gcc/14.2.0 python/3.12.3 cuda/12.8.1
source ~/.venv/bin/activate

export PYTHONUNBUFFERED=1

CSV_FILES=$(ls "${DATA_DIR}"/*.csv 2>/dev/null)
if [ -z "${CSV_FILES}" ]; then
    echo "ERROR: no CSV files found in DATA_DIR=${DATA_DIR}"
    exit 1
fi

ARGS="--csv ${CSV_FILES} --model ${MODEL}"
ARGS="${ARGS} --hidden ${HIDDEN} --layers ${LAYERS} --dropout ${DROPOUT}"
ARGS="${ARGS} --lr ${LR} --batch ${BATCH} --epochs ${EPOCHS} --patience ${PATIENCE}"
ARGS="${ARGS} --rf-search ${RF_SEARCH} --rf-max-samples ${RF_MAX_SAMPLES}"
ARGS="${ARGS} --min-trip-len ${MIN_TRIP_LEN}"
ARGS="${ARGS} --cache-dir ${CACHE_DIR} --output-dir ${OUT_DIR}"
python -u train.py ${ARGS}
EXIT_CODE=$?

echo ""
echo "Finished : $(date)"
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
