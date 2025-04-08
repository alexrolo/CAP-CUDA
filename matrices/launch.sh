sbatch --wrap="./multiply_matrices_cuda 16384 16384 128" \
  --job-name=multiply_matrices_cuda \
  --output=logs/multiply_matrices_cuda$(date +%Y-%m-%d_%H-%M-%S).log \
  --error=logs/multiply_matrices_cuda$(date +%Y-%m-%d_%H-%M-%S).log \
  --time=00:10:00 \
  --partition=cola-gpu \
  --gres=gpu:A100
