#!/bin/bash
#SBATCH --job-name=canon-gemma4-tulu3-lrsweep
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd /scratch/gpfs/ARORA/xd7812/canon
source .venv-gemma4/bin/activate

export HF_HOME=/scratch/gpfs/ARORA/xd7812/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/gpfs/ARORA/xd7812/canon:${PYTHONPATH:-}

mkdir -p logs

# LR sweep on `original` canon (baseline) at LR != 5e-6 to test whether the
# Phase 2 main result lives in the "right" LR regime for canon to surface.
# Existing 5e-6 results are orig_s1/orig_s2 under
# results/tulu3_sft/gemma4_e2b_base/lr5e-6_steps1000/.
#
# idx       0          1          2          3
# lr        2e-6       2e-6       1e-5       1e-5
# seed      1          2          1          2

LRS=(2e-6 2e-6 1e-5 1e-5)
SEEDS=(1   2   1    2  )
TAGS=(orig_lr2e-6_s1 orig_lr2e-6_s2 orig_lr1e-5_s1 orig_lr1e-5_s2)

I=$SLURM_ARRAY_TASK_ID
LR=${LRS[$I]}
SEED=${SEEDS[$I]}
TAG=${TAGS[$I]}

# Output dir naming carries the LR explicitly to keep this sweep separate from
# the main Phase 2 lr5e-6 results.
LR_TAG=$(echo "$LR" | tr '+' '')
SFT_DIR=results/tulu3_sft/gemma4_e2b_base/lr_sweep/${TAG}
IFEVAL_DIR=results/ifeval/gemma4_e2b_base_tulu3sft_lrsweep/${TAG}

python scripts/tulu3_full_finetune.py \
  --model-path /scratch/gpfs/ARORA/xd7812/models/gemma-4-E2B \
  --out-dir "$SFT_DIR" \
  --canonicalizer original \
  --dataset allenai/tulu-3-sft-mixture \
  --lr "$LR" \
  --lr-scheduler-type cosine \
  --warmup-ratio 0.03 \
  --max-steps 1000 \
  --train-samples 8192 \
  --eval-samples 256 \
  --max-length 2048 \
  --seed "$SEED" \
  --gradient-accumulation-steps 8 \
  --logging-steps 25 \
  --eval-strategy-mode steps \
  --eval-steps-interval 250 \
  --eval-on-start \
  --save-final-model \
  --freeze-vision-audio \
  --freeze-per-layer-emb

python scripts/ifeval_generation_eval.py \
  --model-path "$SFT_DIR/final_model" \
  --out-dir "$IFEVAL_DIR" \
  --batch-size 4 \
  --max-new-tokens 1024 \
  --max-input-tokens 1024 \
  --seed "$SEED"
