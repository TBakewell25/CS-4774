#!/bin/bash
#SBATCH --job-name=train_rnn
#SBATCH --output=logs/rnn_%j.out
#SBATCH --error=logs/rnn_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# ── optional args (override at submission time with --export) ─────────────────
#   MAX_CYCLES  : subsample to N cycles (leave unset to use all data)
#   HIDDEN_DIM  : RNN hidden state size          (default 64)
#   LR          : Adam learning rate             (default 1e-3)
#   EPOCHS      : max training epochs            (default 20)
#   BATCH_SIZE  : batch size                     (default 64)
#   PATIENCE    : early-stopping patience        (default 5)
#   OUT_DIR     : where to write outputs         (default: results/<job-id>)

OUT_DIR="${OUT_DIR:-results/${SLURM_JOB_ID}}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
LR="${LR:-1e-3}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PATIENCE="${PATIENCE:-10}"

mkdir -p logs "${OUT_DIR}"

echo "=========================================="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "Started    : $(date)"
echo "GPU        : ${CUDA_VISIBLE_DEVICES:-<auto>}"
echo "OUT_DIR    : ${OUT_DIR}"
echo "HIDDEN_DIM : ${HIDDEN_DIM}"
echo "LR         : ${LR}"
echo "EPOCHS     : ${EPOCHS}"
echo "BATCH_SIZE : ${BATCH_SIZE}"
echo "PATIENCE   : ${PATIENCE}"
echo "MAX_CYCLES : ${MAX_CYCLES:-<all>}"
echo "=========================================="

# ── environment ───────────────────────────────────────────────────────────────
module load gcc/14.2.0 python/3.12.3 cuda/12.8.1
source ~/.venv/bin/activate

# ── build argument list ───────────────────────────────────────────────────────
ARGS="--hidden-dim ${HIDDEN_DIM} --lr ${LR} --epochs ${EPOCHS}"
ARGS="${ARGS} --batch-size ${BATCH_SIZE} --patience ${PATIENCE}"
ARGS="${ARGS} --out-dir ${OUT_DIR}"
if [ -n "${MAX_CYCLES}" ]; then
    ARGS="${ARGS} --max-cycles ${MAX_CYCLES}"
fi

# ── run ───────────────────────────────────────────────────────────────────────
python3 train_rnn.py ${ARGS}
EXIT_CODE=$?

echo ""
echo "Finished : $(date)"
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
