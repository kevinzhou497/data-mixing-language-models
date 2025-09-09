#!/bin/bash
#PBS -lwalltime=12:00:00
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=1
#PBS -o /gpfs/home/klz24/data-mixing-language-models/job_o
#PBS -e  /gpfs/home/klz24/data-mixing-language-models/job_e
module purge
module load Python/3.10.4-GCCcore-11.3.0

cd $HOME/data-mixing-language-models
echo "Current dir: $(pwd)"

echo "Activating conda..."
source myenv310/bin/activate

nvidia-smi || echo "nvidia-smi failed"

python repeat_aware_docs.py -d HuggingFaceFW/fineweb -c sample-10BT --subsample_factor 4 --dataset_split "train" -o data/finewebtext/subsample_4_docs > "fineweb_repeat_4_docs.log" 2>&1

deactivate
module purge
