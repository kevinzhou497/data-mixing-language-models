#!/bin/bash
#PBS -lwalltime=2:00:00
#PBS -l select=1:ncpus=4:mem=8gb:ngpus=1:gpu_type=L40S
#PBS -o /rds/general/user/klz24/thesis/job_o
#PBS -e  /rds/general/user/klz24/thesis/job_e
module purge
module load tools/prod
module load Python/3.10.4-GCCcore-11.3.0

cd $HOME/thesis
source myenv310/bin/activate

nvidia-smi
CUDA_VISIBLE_DEVICES=0

python data_retrieval.py -d HuggingFaceFW/fineweb -c sample-10BT -t 100000000 --num_shards 5 -o data/finewebtext --no_val > "data_retrieval.log" 2>&1

deactivate
module purge