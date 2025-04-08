sbatch --wrap="./equations_cuda 8" \
  --job-name=equations_cuda \
  --output=logs/equations_cuda$(date +%Y-%m-%d_%H-%M-%S).log \
  --error=logs/equations_cuda$(date +%Y-%m-%d_%H-%M-%S).log \
  --time=00:10:00 \
  --partition=cola-gpu \
  --gres=gpu:A100
