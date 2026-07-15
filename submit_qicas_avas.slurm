#!/bin/bash
#SBATCH --job-name=qicas_avas
#SBATCH --account=hpc-prf-qehpc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-15
#SBATCH --output=logs_avas/qicas_avas_%A_%a.out
#SBATCH --error=logs_avas/qicas_avas_%A_%a.err

# ── Environment ───────────────────────────────────────────────────────────
source ~/.block2_fix/block2_env.sh

# ── Working directory ─────────────────────────────────────────────────────
WORKDIR=~/activeml/qio/QIO-master/examples_benchmark/pilot/qicas16
mkdir -p "$WORKDIR/results_avas"
mkdir -p "$WORKDIR/logs_avas"
cd "$WORKDIR"

echo "AVAS-QICAS system index $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"

python qicas_avas_16systems.py \
    --system_index $SLURM_ARRAY_TASK_ID \
    --M 100 \
    --sweeps 30 \
    --threshold 0.1 \
    --out_dir results_avas

echo "Finished at $(date)"
