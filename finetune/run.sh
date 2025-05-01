#!/bin/bash
#SBATCH --job-name=finetune_qwen
#SBATCH --output=slurm_output/output_qwen_finetune_full.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB
#SBATCH --constraint=48GBgpu

source /nlp/data/jsq/thesis/bin/activate

module load cuda/11.7

nvidia-smi
nvcc --version
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('PyTorch Version:', torch.__version__)"
python --version

huggingface-cli login --token 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
wandb login --relogin 3449e394fda9eacb21f123572143a9c0c6dd3069

export HF_HOME=/nlp/data/huggingface_cache

python finetune.py