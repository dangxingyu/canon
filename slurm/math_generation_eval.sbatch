#!/bin/bash
#SBATCH --job-name=canon-math-gen
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
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

CANONICALIZERS=(original combined original combined)
SEEDS=(1 1 2 2)

CANONICALIZER=${CANONICALIZERS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MODEL_DIR=results/math_full_finetune_for_generation/qwen3_0_6b/lr2e-5_steps1000/${CANONICALIZER}_seed${SEED}/final_model
OUT_DIR=results/math_generation_eval/qwen3_0_6b/lr2e-5_steps1000_math500/${CANONICALIZER}_seed${SEED}

python scripts/math_generation_eval.py \
  --model-path "$MODEL_DIR" \
  --out-dir "$OUT_DIR" \
  --max-samples -1 \
  --batch-size 8 \
  --max-input-tokens 1024 \
  --max-new-tokens 256 \
  --seed "$SEED" \
  --torch-dtype bfloat16
