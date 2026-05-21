#!/bin/bash
#SBATCH --job-name=canon-gsm8k-lora
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
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

CANONICALIZERS=(original vo_norm mlp_norm combined)
CANONICALIZER=${CANONICALIZERS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR=results/gsm8k_lora_smoke/qwen3_0_6b/r8_lr2e-4_steps80/${CANONICALIZER}_seed1

python scripts/gsm8k_lora_smoke.py \
  --model-path /scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B \
  --out-dir "$OUT_DIR" \
  --canonicalizer "$CANONICALIZER" \
  --rank 8 \
  --lora-alpha 16 \
  --lr 2e-4 \
  --max-steps 80 \
  --train-samples 512 \
  --eval-samples 128 \
  --max-length 512 \
  --seed 1
