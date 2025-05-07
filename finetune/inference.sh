#!/bin/bash
#SBATCH --job-name=inference_english
#SBATCH --output=finetune/slurm_output/output_inference.txt
#SBATCH --partition=p_nlp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256GB
#SBATCH --constraint=48GBgpu
#SBATCH --time=24:00:00

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

python -m finetune.inference --model_dir english_translated_llama_final --dataset cvqa --num_samples 3