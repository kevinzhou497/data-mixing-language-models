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
python repeat_aware_documents_pubmed.py \
    -d "ncbi/pubmed" \
    -o "data/pubmed" \
    -s 500000000 \
    --shuffle_seed 42 \
    --subsample 1 \
    > "pubmed_repeat_1_docs.log" 2>&1
deactivate
module purge
