#!/bin/bash
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=2
#PBS -o /gpfs/home/klz24/data-mixing-language-models/job_o
#PBS -e /gpfs/home/klz24/data-mixing-language-models/job_e

module purge
module load Python/3.10.4-GCCcore-11.3.0

cd $HOME/data-mixing-language-models
echo "Current dir: $(pwd)"

echo "Activating conda..."
source myenv310/bin/activate

nvidia-smi || echo "nvidia-smi failed"

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4

MIX_RATIOS=(1.0)
LEARNING_RATES=(0.001)
ITERATIONS=(14280)
SAMPLE=8
HQ_DATASET="wikipedia"
MODEL_PARAMS="1B"

GLOBAL_BATCH_SIZE=128
DEVICE_BATCH_SIZE=32

for LR in "${LEARNING_RATES[@]}"; do
  for MIX in "${MIX_RATIOS[@]}"; do
    for NUM_ITERATIONS in "${ITERATIONS[@]}"; do

      LOGDIR="logs/${HQ_DATASET}/${NUM_ITERATIONS}iters/mix${MIX}_lr${LR}_${MODEL_PARAMS}_${SAMPLE}"
      mkdir -p "$LOGDIR"

      echo "Starting run: MIX=$MIX LR=$LR ITERS=$NUM_ITERATIONS"
      echo "Global batch size: $GLOBAL_BATCH_SIZE"
      echo "Device batch size: $DEVICE_BATCH_SIZE"

      torchrun --standalone --nproc_per_node=2 train_gpt_1B.py \
        --train_bin_primary "data/finewebtext/train_*.bin" \
        --train_bin_secondary "data/wikitext/subsample_${SAMPLE}_docs/train_*.bin" \
        --mixing_ratio "$MIX" \
        --val_bin "data/wikitext/subsample_1_docs/validation_*.bin" \
        --learning_rate "$LR" \
        --seed 42 \
        --params "$MODEL_PARAMS" \
        --subsample "$SAMPLE" \
        --num_iterations "$NUM_ITERATIONS" \
        --hq_dataset "$HQ_DATASET" \
        --logdir "$LOGDIR" \
        --global_batch_size "$GLOBAL_BATCH_SIZE" \
        --device_batch_size "$DEVICE_BATCH_SIZE" \
        --grad_checkpointing \
        > "$LOGDIR/train.log" 2>&1

    done
  done
done

deactivate
module purge