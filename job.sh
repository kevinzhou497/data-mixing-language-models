#!/bin/bash
#PBS -lwalltime=12:00:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=L40S
#PBS -o /rds/general/user/klz24/home/thesis/job_o
#PBS -e  /rds/general/user/klz24/home/thesis/job_e
module purge
module load tools/prod
module load Python/3.10.4-GCCcore-11.3.0

cd $HOME/thesis
source myenv310/bin/activate

nvidia-smi
CUDA_VISIBLE_DEVICES=0

MIX_RATIOS=(0.33)
LEARNING_RATES=(0.001 0.002 0.003 0.004 0.005)
NUM_ITERATIONS=29074

for MIX in "${MIX_RATIOS[@]}"; do
  for LR in "${LEARNING_RATES[@]}"; do
    LOGDIR="logs/${NUM_ITERATIONS}iters/mix${MIX}_lr${LR}"
    mkdir -p "$LOGDIR"
    torchrun --standalone --nproc_per_node=1 train_gpt.py \
      --train_bin_primary "data/finewebtext/train_*.bin" \
      --train_bin_secondary "data/wiki/train_*.bin" \
      --mixing_ratio $MIX \
      --val_bin "data/wiki/val_*.bin" \
      --learning_rate $LR \
      --seed 42 \
      --num_iterations $NUM_ITERATIONS \
      --logdir "$LOGDIR" > "$LOGDIR/train.log" 2>&1
  done
done

deactivate
module purge
