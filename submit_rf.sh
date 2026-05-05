#!/bin/bash
#SBATCH --job-name=train_rf
#SBATCH --output=logs/rf_%j.out
#SBATCH --error=logs/rf_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=cpu

# ── optional args (override at submission time with --export) ─────────────────
#   MAX_ROWS  : subsample size (leave unset to use all data)
#   N_ITER    : number of hyperparameter trials (default 10)
#   OUT_DIR   : where to write outputs (default: results/<job-id>)

OUT_DIR="${OUT_DIR:-results/${SLURM_JOB_ID}}"
N_ITER="${N_ITER:-10}"

mkdir -p logs "${OUT_DIR}"

echo "=========================================="
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : $(hostname)"
echo "Started  : $(date)"
echo "OUT_DIR  : ${OUT_DIR}"
echo "N_ITER   : ${N_ITER}"
echo "MAX_ROWS : ${MAX_ROWS:-<all>}"
echo "=========================================="

# ── environment ───────────────────────────────────────────────────────────────
# Adjust the module name / conda env to match your cluster setup.
# module load python/3.11          # uncomment if modules are used
# source ~/.venv/bin/activate      # uncomment if using a venv

# ── build argument list ───────────────────────────────────────────────────────
ARGS="--n-iter ${N_ITER} --out-dir ${OUT_DIR}"
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
