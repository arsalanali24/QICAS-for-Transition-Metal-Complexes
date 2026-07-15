#!/bin/bash
#SBATCH --job-name=qicas16
#SBATCH --account=hpc-prf-qehpc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/qicas16_%A_%a.out
#SBATCH --error=logs/qicas16_%A_%a.err

# ── Environment ──────────────────────────────────────────────────────────
source ~/.block2_fix/block2_env.sh

# ── Working directory ─────────────────────────────────────────────────────
WORKDIR=~/activeml/qio/QIO-master/examples_benchmark/pilot/qicas16
mkdir -p "$WORKDIR/results"
mkdir -p "$WORKDIR/logs"
cd "$WORKDIR"

# ── Run ───────────────────────────────────────────────────────────────────
echo "Starting system index $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
echo "System: $(python -c "
from qicas_casscf_16systems import ALL_SYSTEM_NAMES
print(ALL_SYSTEM_NAMES[$SLURM_ARRAY_TASK_ID])
")"

python qicas_casscf_16systems.py \
    --system_index $SLURM_ARRAY_TASK_ID \
    --M 100 \
    --sweeps 30 \
    --out_dir results

echo "Finished at $(date)"
