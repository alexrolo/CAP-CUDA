sbatch --wrap="./comparer_cuda" \
  --job-name=s3_eq_gpu \
  --output=logs/equations_gpu_$(date +%Y-%m-%d_%H-%M-%S).log \
  --error=logs/equations_gpu_$(date +%Y-%m-%d_%H-%M-%S).log \
  --time=12:00:00 \
  --partition=cola-gpu \
  --gres=gpu:A100
