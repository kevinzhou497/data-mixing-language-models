#!/bin/bash
#PBS -lwalltime=24:00:00
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1
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

MIX_RATIOS=(0.75) 

LEARNING_RATES=(0.001)
ITERATIONS=(58566)
SAMPLE=2
HQ_DATASET="pubmed"
HQ_directory="pubmed"
MODEL_PARAMS="345M" 

for LR in "${LEARNING_RATES[@]}"; do
  for MIX in "${MIX_RATIOS[@]}"; do
    for NUM_ITERATIONS in "${ITERATIONS[@]}"; do
      LOGDIR="logs/${HQ_DATASET}/${NUM_ITERATIONS}iters/mix${MIX}_lr${LR}_${MODEL_PARAMS}_${SAMPLE}"
      mkdir -p "$LOGDIR"
      torchrun --standalone --nproc_per_node=1 train_gpt.py \
        --train_bin_primary "data/finewebtext/subsample_${SAMPLE}_docs/train_*.bin" \
        --train_bin_secondary "data/${HQ_directory}/train_subsamples/subsample_${SAMPLE}_docs/train_*.bin" \
        --mixing_ratio $MIX \
        --val_bin "data/${HQ_directory}/val_200K/validation_*.bin" \
        --learning_rate $LR \
        --seed 42 \
        --params $MODEL_PARAMS \
        --subsample $SAMPLE \
        --num_iterations $NUM_ITERATIONS \
        --hq_dataset $HQ_DATASET \
        --logdir "$LOGDIR" > "$LOGDIR/train.log" 2>&1
    done
  done
done

deactivate
module purge
