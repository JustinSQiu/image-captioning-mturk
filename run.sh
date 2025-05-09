#!/bin/bash
#SBATCH --job-name=synthetic_data
#SBATCH --output=slurm_output/output_synthetic.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --constraint=48GBgpu

source /nlp/data/jsq/thesis/bin/activate

module load cuda/11.7

nvidia-smi
nvcc --version
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('PyTorch Version:', torch.__version__)"
python --version

huggingface-cli login --token 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

python -m pipeline.gpt4o_synthetic_data.py