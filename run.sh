#!/bin/bash
#SBATCH --job-name=transcribe
#SBATCH --output=slurm_output/output_transcription_test.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --constraint=48GBgpu
#SBATCH --time=1:00:00

source /nlp/data/jsq/thesis/bin/activate

module load cuda/11.7

nvidia-smi
nvcc --version
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('PyTorch Version:', torch.__version__)"
python --version

huggingface-cli login --token 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

python pipeline/1_transcription.py
# python test_transcription.py