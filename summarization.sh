#!/bin/bash
#SBATCH --job-name=summarization
#SBATCH --output=slurm_output/summarization.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --constraint=48GBgpu

source /nlp/data/jsq/thesis/bin/activate

python -m pipeline.6_extract_qa