#!/bin/bash
#SBATCH --job-name=canon-full-ft-lr
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
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

CANONICALIZERS=(original combined original combined original combined original combined)
SEEDS=(1 1 2 2 1 1 2 2)
LRS=(1e-5 1e-5 1e-5 1e-5 5e-5 5e-5 5e-5 5e-5)
LR_LABELS=(lr1e-5 lr1e-5 lr1e-5 lr1e-5 lr5e-5 lr5e-5 lr5e-5 lr5e-5)

CANONICALIZER=${CANONICALIZERS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
LR_LABEL=${LR_LABELS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR=results/gsm8k_full_finetune/qwen3_0_6b/${LR_LABEL}_steps500/${CANONICALIZER}_seed${SEED}

python scripts/gsm8k_full_finetune.py \
  --model-path /scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B \
  --out-dir "$OUT_DIR" \
  --canonicalizer "$CANONICALIZER" \
  --lr "$LR" \
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
