# Generated SLURM script

#!/bin/bash
#SBATCH --job-name=train_rf
#SBATCH --output=logs/rf_%j.out
#SBATCH --error=logs/rf_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=cpu

# ── configuration ─────────────────────────────────────────────────────────────
# Required: set DATA_DIR to the folder containing your parsed CSVs.
#   sbatch --export=ALL,DATA_DIR=/path/to/parsed submit_rf.sh
#
# Optional overrides (pass via --export):
#   MAX_ROWS  : subsample to N total rows    (default: all)
#   N_ITER    : random-search iterations     (default: 10)
#   OUT_DIR   : output directory             (default: results/<job-id>)

if [ -z "${DATA_DIR}" ]; then
    echo "ERROR: DATA_DIR is not set. Pass it with --export=ALL,DATA_DIR=/path/to/parsed"
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${OUT_DIR:-results/${SLURM_JOB_ID}}"
N_ITER="${N_ITER:-10}"

mkdir -p logs "${OUT_DIR}"

echo "=========================================="
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Started   : $(date)"
echo "DATA_DIR  : ${DATA_DIR}"
echo "OUT_DIR   : ${OUT_DIR}"
echo "N_ITER    : ${N_ITER}"
echo "MAX_ROWS  : ${MAX_ROWS:-<all>}"
echo "=========================================="

# ── environment ───────────────────────────────────────────────────────────────
module load gcc/14.2.0 python/3.12.3
source ~/.venv/bin/activate

# ── build CSV list from DATA_DIR ──────────────────────────────────────────────
CSV_FILES=$(ls "${DATA_DIR}"/*.csv 2>/dev/null)
if [ -z "${CSV_FILES}" ]; then
    echo "ERROR: no CSV files found in DATA_DIR=${DATA_DIR}"
    exit 1
fi

# ── build argument list ───────────────────────────────────────────────────────
ARGS="--csv ${CSV_FILES}"
ARGS="${ARGS} --n-iter ${N_ITER}"
ARGS="${ARGS} --out-dir ${OUT_DIR}"
if [ -n "${MAX_ROWS}" ]; then
    ARGS="${ARGS} --max-rows ${MAX_ROWS}"
fi

# ── run ───────────────────────────────────────────────────────────────────────
python3 train_rf.py ${ARGS}
EXIT_CODE=$?

echo ""
echo "Finished : $(date)"
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
