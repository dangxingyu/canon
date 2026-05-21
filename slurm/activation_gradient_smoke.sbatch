#!/bin/bash
#SBATCH --job-name=canon-actgrad
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /scratch/gpfs/ARORA/xd7812/canon
source /usr/licensed/anaconda3/2025.6/etc/profile.d/conda.sh
conda activate rllm

export HF_HOME=/scratch/gpfs/ARORA/xd7812/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/gpfs/ARORA/xd7812/canon:${PYTHONPATH:-}

mkdir -p logs

python scripts/activation_gradient_smoke.py \
  --model-path /scratch/gpfs/ARORA/xd7812/models/Qwen3-0.6B \
  --out-dir results/activation_gradient_smoke/qwen3_0_6b_bf16_s8_len256 \
  --dtype bfloat16 \
  --max-samples 8 \
  --max-length 256 \
  --seed 1
