sbatch --wrap="./cuda_mul_benchmark" \
  --job-name=cuda_mul_benchmark \
  --output=logs/cuda_mul_benchmark$(date +%Y-%m-%d_%H-%M-%S).log \
  --error=logs/cuda_mul_benchmark_error_$(date +%Y-%m-%d_%H-%M-%S).log \
  --partition=cola-gpu \
  --gres=gpu:A100
