sbatch --wrap="./equations_cuda 750 128" \
  --job-name=eq_cuda \
  --output=logs/equations_cuda_$(date +%Y-%m-%d_%H-%M-%S).log \
  --error=logs/equations_cuda_$(date +%Y-%m-%d_%H-%M-%S).log \
  --time=06:00:00 \
  --partition=cola-gpu \
  --gres=gpu:A100
