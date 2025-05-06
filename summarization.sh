#!/bin/bash
#SBATCH --job-name=summarization
#SBATCH --output=slurm_output/summarization.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --constraint=48GBgpu

source /nlp/data/jsq/thesis/bin/activate

# cd /home1/j/jsq/dev/image-captioning-mturk

python -m pipeline.5a_summarization

python -m pipeline.5b_summarization_english

python -m pipeline.5c_summarization_native_nonnative