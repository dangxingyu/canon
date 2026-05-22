#!/bin/bash
#SBATCH --job-name=canon-math-abl-0.6b
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-3%4
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

# 0.6B mirror of math_ablation_vo_mlp_1_7b.sbatch. Isolates V/O vs MLP halves
# at small saturated scale, where the SPD trajectory's combined-vs-original
# delta was already ~1e-4 -- here we see if either half alone can explain it.

CANONS=(vo_norm vo_norm mlp_norm mlp_norm)
SEEDS=( 1       2       1        2       )
TAGS=(  vo_norm_s1 vo_norm_s2 mlp_norm_s1 mlp_norm_s2)

I=$SLURM_ARRAY_TASK_ID
CANON=${CANONS[$I]}
SEED=${SEEDS[$I]}
TAG=${TAGS[$I]}

OUT_DIR=results/math_ablation_vo_mlp/qwen3_0_6b/lr2e-5_steps1000/${TAG}

python scripts/math_full_finetune.py \
  --model-path /scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B \
  --out-dir "$OUT_DIR" \
  --canonicalizer "$CANON" \
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
  --eval-on-start
