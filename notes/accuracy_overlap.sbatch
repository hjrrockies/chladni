#!/bin/bash
#SBATCH --job-name=chladni_accuracy_overlap
#SBATCH --array=0-50
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --partition=normal_q
#SBATCH --account=siallocation
#SBATCH --output=accuracy_overlap_%A_%a.out

# notes/accuracy_studies.py "overlap" on ARC (2026-11-05): per douse reference shape and hard
# ratio shape, the cross-level Rellich Gram of adjacent levels within 1e-2 relative, at prec 1e-8,
# 1e-10 and 1e-12: entanglement (falls with prec) against the gap bias (flat in prec). The array
# size must equal `python accuracy_studies.py ntasks overlap` (51).
#
# Its OWN checkout, so pulling it cannot touch chladni_hee under running FDref/grad jobs. From
# ~/projects/thesis:
#   git -C chladni fetch origin
#   git -C chladni worktree add ../chladni_c0 origin/hadamard-error-estimate
# Submit FROM its notes directory (the logs land there):
#   cd ~/projects/thesis/chladni_c0/notes && sbatch --mail-user=$DOUSE_SLURM_MAIL --mail-type=END,FAIL accuracy_overlap.sbatch
# Then:
#   ~/projects/thesis/douse/.venv/bin/python accuracy_studies.py analyze_overlap accuracy_overlap_out
set -euo pipefail
ROOT=${THESIS_ROOT:-$HOME/projects/thesis}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$ROOT/douse/.venv/bin/python" "$ROOT/chladni_c0/notes/accuracy_studies.py" task "$SLURM_ARRAY_TASK_ID" \
    --tip "$ROOT/chladni_c0/src" --table overlap \
    --out "$ROOT/chladni_c0/notes/accuracy_overlap_out"
