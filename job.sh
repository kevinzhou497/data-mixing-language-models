#!/bin/bash
#PBS -lwalltime=36:00:00
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

MIX_RATIOS=(0.55 0.65) 
MIXING_RATIOS_LIST=(
  "0.40 0.30 0.30" 
  "0.45 0.275 0.275"
)
LEARNING_RATES=(0.00141)
ITERATIONS=(115665)
SAMPLE=1
HQ_DATASET="wikipedia"
HQ_directory="wikitext"
HQ_DATASET_2="pubmed"
HQ_directory_2="pubmed"
MODEL_PARAMS="124M" 
logging_hq="wikipedia_pubmed"
for LR in "${LEARNING_RATES[@]}"; do
  for MIX in "${MIXING_RATIOS_LIST[@]}"; do
    for NUM_ITERATIONS in "${ITERATIONS[@]}"; do
      MIX_NAME=$(echo "$MIX" | tr ' ' '_') 
      LOGDIR="logs/${HQ_DATASET}_${HQ_DATASET_2}/${NUM_ITERATIONS}iters/mix${MIX_NAME}_lr${LR}_${MODEL_PARAMS}_${SAMPLE}"
      mkdir -p "$LOGDIR"
      torchrun --standalone --nproc_per_node=1 train_gpt.py \
        --train_bin_primary "data/finewebtext/train_*.bin" \
        --train_bin_secondary "data/${HQ_directory}/subsample_${SAMPLE}_docs/train_*.bin" \
        --train_bin_third "data/${HQ_directory_2}/train_subsamples/subsample_${SAMPLE}_docs/shard_*.bin" \
        --mixing_ratios $MIX \
        --val_bin "data/${HQ_directory}/subsample_1_docs/validation_*.bin" \
        --val_bin_2 "data/${HQ_directory_2}/val_200K/shard_*.bin" \
        --learning_rate $LR \
        --seed 42 \
        --params $MODEL_PARAMS \
        --subsample $SAMPLE \
        --num_iterations $NUM_ITERATIONS \
        --hq_dataset $logging_hq \
        --logdir "$LOGDIR" > "$LOGDIR/train.log" 2>&1
    done
  done
done

deactivate
module purge
