#!/bin/bash
#SBATCH --job-name=canon-gemma4-tulu3-mid
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

# Mid-training canon on Gemma-4 E2B + Tülu-3. Phase 1 showed mid > init on
# Qwen3-0.6B but init > mid on Qwen3-1.7B. Gemma-4 is bigger and harder to
# finetune, so we expect either init ≥ mid (matching 1.7B) or potentially a
# new regime where re-applying orth gauge mid-training does help (because
# Gemma's grads are large and weights drift fast).
#
# idx       0                   1                   2                   3
# init      combined_orth       combined_orth       combined_orth       combined_orth
# mid       combined_orth       combined_orth       combined_orth       combined_orth
# mode      lazy                lazy                reset               reset
# seed      1                   2                   1                   2

MODES=(lazy lazy reset reset)
SEEDS=(1    2    1     2    )
TAGS=(comb_orth_lazy500_s1 comb_orth_lazy500_s2 comb_orth_reset500_s1 comb_orth_reset500_s2)

I=$SLURM_ARRAY_TASK_ID
MODE=${MODES[$I]}
SEED=${SEEDS[$I]}
TAG=${TAGS[$I]}

SFT_DIR=results/tulu3_sft/gemma4_e2b_base/lr5e-6_steps1000_midcanon/${TAG}
IFEVAL_DIR=results/ifeval/gemma4_e2b_base_tulu3sft_midcanon/${TAG}

python scripts/tulu3_full_finetune.py \
  --model-path /scratch/gpfs/ARORA/xd7812/models/gemma-4-E2B \
  --out-dir "$SFT_DIR" \
  --canonicalizer combined_orth \
  --mid-canon-steps 500 \
  --mid-canon-mode "$MODE" \
  --mid-canon-name combined_orth \
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

python scripts/ifeval_generation_eval.py \
  --model-path "$SFT_DIR/final_model" \
  --out-dir "$IFEVAL_DIR" \
  --batch-size 4 \
  --max-new-tokens 1024 \
  --max-input-tokens 1024 \
  --seed "$SEED"
