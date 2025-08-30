#!/bin/bash
#PBS -lwalltime=8:00:00
#PBS -l select=1:ncpus=8:mem=24gb:ngpus=1
#PBS -o /gpfs/home/klz24/data-mixing-language-models/job_o
#PBS -e  /gpfs/home/klz24/data-mixing-language-models/job_e

echo "Job started at $(date)"
module purge
module load Python/3.10.4-GCCcore-11.3.0

cd $HOME/data-mixing-language-models
echo "Current dir: $(pwd)"

echo "Activating conda..."
source myenv310/bin/activate

nvidia-smi || echo "nvidia-smi failed"

CUDA_VISIBLE_DEVICES=0


python -u data_retrieval.py -d wikitext -c wikitext-103-raw-v1 --num_shards 1 -o data/wikitext --dataset_split "validation" > "data_retrieval_wiki_val.log" 2>&1

deactivate
module purge