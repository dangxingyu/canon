#!/bin/bash
#SBATCH --job-name=canon-math-traj-1.7b
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd /scratch/gpfs/ARORA/xd7812/canon
source /usr/licensed/anaconda3/2025.6/etc/profile.d/conda.sh
conda activate rllm

export HF_HOME=/scratch/gpfs/ARORA/xd7812/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/gpfs/ARORA/xd7812/canon:${PYTHONPATH:-}

mkdir -p logs

# idx       0          1          2          3          4          5          6          7
# init      original   original   combined   combined   combined   combined   combined   combined
# mid       (none)     (none)     (none)     (none)     lazy       lazy       reset      reset
# seed      1          2          1          2          1          2          1          2

INIT_CANONS=(original original combined combined combined combined combined combined)
MID_STEPS=( ""       ""       ""       ""       500      500      500      500     )
MID_MODES=( lazy     lazy     lazy     lazy     lazy     lazy     reset    reset   )
SEEDS=(     1        2        1        2        1        2        1        2       )
TAGS=(      orig_s1  orig_s2  comb_s1  comb_s2  comb_lazy500_s1  comb_lazy500_s2  comb_reset500_s1  comb_reset500_s2)

I=$SLURM_ARRAY_TASK_ID
INIT=${INIT_CANONS[$I]}
MID_STEP=${MID_STEPS[$I]}
MID_MODE=${MID_MODES[$I]}
SEED=${SEEDS[$I]}
TAG=${TAGS[$I]}

OUT_DIR=results/math_trajectory/qwen3_1_7b_base/lr2e-5_steps1000/${TAG}

EXTRA_ARGS=()
if [ -n "$MID_STEP" ]; then
  EXTRA_ARGS+=(--mid-canon-steps "$MID_STEP" --mid-canon-mode "$MID_MODE" --mid-canon-name combined)
fi

python scripts/math_full_finetune.py \
  --model-path /scratch/gpfs/ARORA/xd7812/models/Qwen3-1.7B-Base \
  --out-dir "$OUT_DIR" \
  --canonicalizer "$INIT" \
  --eval-source math500 \
  --lr 2e-5 \
  --lr-scheduler-type constant_with_warmup \
  --warmup-ratio 0.03 \
  --max-steps 1000 \
  --max-length 1024 \
  --min-answer-tokens 128 \
  --seed "$SEED" \
  --gradient-accumulation-steps 8 \
  --logging-steps 25 \
  --eval-strategy-mode steps \
  --eval-steps-interval 250 \
  --eval-on-start \
  "${EXTRA_ARGS[@]}"
