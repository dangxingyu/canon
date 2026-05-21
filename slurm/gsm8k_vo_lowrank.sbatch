#!/bin/bash
#SBATCH --job-name=canon-vo-lowrank
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
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
RANKS=(2 2 2 4 4 4)

CANONICALIZER=${CANONICALIZERS[$SLURM_ARRAY_TASK_ID]}
RANK=${RANKS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR=results/gsm8k_vo_lowrank/qwen3_0_6b/r${RANK}_lr3e-4_const_steps160/${CANONICALIZER}_seed1

python scripts/gsm8k_lora_smoke.py \
  --model-path /scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B \
  --out-dir "$OUT_DIR" \
  --canonicalizer "$CANONICALIZER" \
  --target-modules vo \
  --rank "$RANK" \
  --lora-alpha "$((RANK * 2))" \
  --lr 3e-4 \
  --lr-scheduler-type constant_with_warmup \
  --warmup-ratio 0.03 \
  --max-steps 160 \
  --train-samples 1024 \
  --eval-samples 256 \
  --max-length 512 \
  --seed 1
