#!/bin/bash
#SBATCH --job-name=chladni_accuracy_grad
#SBATCH --array=0-50
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --partition=normal_q
#SBATCH --account=siallocation
#SBATCH --output=accuracy_grad_%A_%a.out

# notes/accuracy_studies.py "grad" on ARC (2026-11-05): per douse reference shape and hard ratio
# shape, the Hadamard vertex gradients at 1e-10 against 1e-12, beside the exact-identity
# residuals. The array size must equal `python accuracy_studies.py ntasks grad` (51).
#
# The same checkout as hadamard_fdref.sbatch, from ~/projects/thesis:
#   git -C chladni fetch origin
#   git -C chladni worktree add ../chladni_hee origin/hadamard-error-estimate
# Submit FROM its notes directory (the logs land there):
#   cd ~/projects/thesis/chladni_hee/notes && sbatch --mail-user=$DOUSE_SLURM_MAIL --mail-type=END,FAIL accuracy_grad.sbatch
# Then:
#   ~/projects/thesis/douse/.venv/bin/python accuracy_studies.py analyze_grad accuracy_grad_out
set -euo pipefail
ROOT=${THESIS_ROOT:-$HOME/projects/thesis}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$ROOT/douse/.venv/bin/python" "$ROOT/chladni_hee/notes/accuracy_studies.py" task "$SLURM_ARRAY_TASK_ID" \
    --tip "$ROOT/chladni_hee/src" --table grad \
    --out "$ROOT/chladni_hee/notes/accuracy_grad_out"
