sbatch --wrap="./comparer" \
  --job-name=s3_eq_cpu \
  --output=logs/equations_cpu_$(date +%Y-%m-%d_%H-%M-%S).log \
  --error=logs/equations_cpu_$(date +%Y-%m-%d_%H-%M-%S).log \
  --time=06:00:00
