#!/bin/bash
#SBATCH --job-name=canon-actgrad-gm
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
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

CANONICALIZERS=(original actgrad_mlp_gmean actgrad_combined_gmean)
CANONICALIZER=${CANONICALIZERS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR=results/gsm8k_actgrad_gmean/qwen3_0_6b/r8_lr2e-4_const_steps160/${CANONICALIZER}_seed1

python scripts/gsm8k_lora_smoke.py \
  --model-path /scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B \
  --out-dir "$OUT_DIR" \
  --canonicalizer "$CANONICALIZER" \
  --target-modules all_linear \
  --rank 8 \
  --lora-alpha 16 \
  --lr 2e-4 \
  --lr-scheduler-type constant_with_warmup \
  --warmup-ratio 0.03 \
  --max-steps 160 \
  --train-samples 1024 \
  --eval-samples 256 \
  --max-length 512 \
  --seed 1 \
  --actgrad-samples 8 \
  --actgrad-max-length 256
