#!/bin/bash
#SBATCH --job-name=transcribe
#SBATCH --output=slurm_output/output_transcription.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --constraint=48GBgpu
#SBATCH --time=3-0

source /nlp/data/jsq/thesis/bin/activate

module load cuda/11.7

nvidia-smi
nvcc --version
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('PyTorch Version:', torch.__version__)"
python --version

huggingface-cli login

python pipeline/1_transcription.py