#!/bin/bash
#PBS -lwalltime=66:00:00
#PBS -l select=1:ncpus=16:mem=80gb:ngpus=1
#PBS -o /gpfs/home/klz24/data-mixing-language-models/job_o
#PBS -e  /gpfs/home/klz24/data-mixing-language-models/job_e
module purge
module load Python/3.10.4-GCCcore-11.3.0

cd $HOME/data-mixing-language-models
echo "Current dir: $(pwd)"

echo "Activating conda..."
source myenv310/bin/activate

nvidia-smi || echo "nvidia-smi failed"

CUDA_VISIBLE_DEVICES=0

MIX_RATIOS=(0.50 0.60 0.67 0.75)
LEARNING_RATES=(0.001 0.000707 0.00141)
NUM_ITERATIONS=116295

for LR in "${LEARNING_RATES[@]}"; do
  for MIX in "${MIX_RATIOS[@]}"; do
    if [[ "$MIX" == "0.50" && "$LR" == "0.001" ]]; then
      continue
    fi

    LOGDIR="logs/${NUM_ITERATIONS}iters/mix${MIX}_lr${LR}"
    mkdir -p "$LOGDIR"
    torchrun --standalone --nproc_per_node=1 train_gpt.py \
      --train_bin_primary "data/finewebtext/train_*.bin" \
      --train_bin_secondary "data/wikitext/train_*.bin" \
      --mixing_ratio $MIX \
      --val_bin "data/wikitext/validation_*.bin" \
      --learning_rate $LR \
      --seed 42 \
      --params "124M" \
      --num_iterations $NUM_ITERATIONS \
      --logdir "$LOGDIR" > "$LOGDIR/train.log" 2>&1
  done
done

deactivate
module purge
