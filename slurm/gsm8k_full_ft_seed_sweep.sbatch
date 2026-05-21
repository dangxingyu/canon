#!/bin/bash
#SBATCH --job-name=canon-full-ft-sd
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-5%6
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

CANONICALIZERS=(original vo_norm combined original vo_norm combined)
SEEDS=(3 3 3 4 4 4)

CANONICALIZER=${CANONICALIZERS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR=results/gsm8k_full_finetune/qwen3_0_6b/lr2e-5_steps500/${CANONICALIZER}_seed${SEED}

python scripts/gsm8k_full_finetune.py \
  --model-path /scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B \
  --out-dir "$OUT_DIR" \
  --canonicalizer "$CANONICALIZER" \
  --lr 2e-5 \
  --weight-decay 0.0 \
  --lr-scheduler-type constant_with_warmup \
  --warmup-ratio 0.03 \
  --max-steps 500 \
  --train-samples 4096 \
  --eval-samples -1 \
  --max-length 512 \
  --gradient-accumulation-steps 8 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --seed "$SEED" \
  --gradient-checkpointing
