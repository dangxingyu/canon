#!/bin/bash
#SBATCH --job-name=canon-gemma4-tulu3
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
source .venv-gemma4/bin/activate

export HF_HOME=/scratch/gpfs/ARORA/xd7812/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/scratch/gpfs/ARORA/xd7812/canon:${PYTHONPATH:-}

mkdir -p logs

# Phase 2 main: Gemma-4 E2B base + Tülu-3 SFT, then IFEval on the saved model.
# 4 canon conditions x 2 seeds = 8 tasks.
#  - original:       baseline, no canon
#  - mlp_norm:       diagonal MLP gauge only (the source of first_grad drop on Qwen3)
#  - vo_orth:        orthogonal V/O gauge only (Gemma-4 compatible due to v_norm)
#  - combined_orth:  vo_orth + mlp_norm

CANONS=(original mlp_norm vo_orth combined_orth original mlp_norm vo_orth combined_orth)
SEEDS=( 1        1        1       1             2        2        2       2            )
TAGS=(  orig_s1  mlp_s1   vo_orth_s1 comb_orth_s1 orig_s2 mlp_s2  vo_orth_s2 comb_orth_s2)

I=$SLURM_ARRAY_TASK_ID
CANON=${CANONS[$I]}
SEED=${SEEDS[$I]}
TAG=${TAGS[$I]}

SFT_DIR=results/tulu3_sft/gemma4_e2b_base/lr5e-6_steps1000/${TAG}
IFEVAL_DIR=results/ifeval/gemma4_e2b_base_tulu3sft/${TAG}

# 1) Tülu-3 SFT
python scripts/tulu3_full_finetune.py \
  --model-path /scratch/gpfs/ARORA/xd7812/models/gemma-4-E2B \
  --out-dir "$SFT_DIR" \
  --canonicalizer "$CANON" \
  --dataset allenai/tulu-3-sft-mixture \
  --lr 5e-6 \
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

# 2) IFEval generation eval on the saved final model
python scripts/ifeval_generation_eval.py \
  --model-path "$SFT_DIR/final_model" \
  --out-dir "$IFEVAL_DIR" \
  --batch-size 4 \
  --max-new-tokens 1024 \
  --max-input-tokens 1024 \
  --seed "$SEED"
