#!/bin/bash
#SBATCH --job-name=chladni_hadamard_fdref
#SBATCH --array=0-59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --partition=normal_q
#SBATCH --account=siallocation
#SBATCH --output=hadamard_fdref_%A_%a.out

# notes/hadamard_hard_shapes.py FDref on ARC (2026-11-05): an FD gradient reference WITH an error
# bar on the ten hard douse ratio shapes, and main's Hadamard on the same shapes.
# Tasks 0-39 are the references, one per (case, direction): one plan fixed at the base shape,
# steps 1e-3, 3e-4, 1e-4, each Richardson-combined from +-h and +-2h at prec 1e-12 (13 solves).
# Tasks 40-59 are the centred Hadamard probes ("Hc") at 1e-10 and 1e-12. The array size must
# equal len(tasks_fdref()).
#
# Once, from ~/projects/thesis:
#   git -C chladni fetch origin
#   git -C chladni worktree add ../chladni_hee origin/hadamard-error-estimate
# Submit FROM the notes directory of that checkout (the logs land there):
#   cd ~/projects/thesis/chladni_hee/notes && sbatch --mail-user=$DOUSE_SLURM_MAIL --mail-type=END,FAIL hadamard_fdref.sbatch
# Then:
#   ~/projects/thesis/douse/.venv/bin/python hadamard_hard_shapes.py analyze_fdref hadamard_fdref_out
set -euo pipefail
ROOT=${THESIS_ROOT:-$HOME/projects/thesis}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$ROOT/douse/.venv/bin/python" "$ROOT/chladni_hee/notes/hadamard_hard_shapes.py" task "$SLURM_ARRAY_TASK_ID" \
    --new "$ROOT/chladni_hee/src" --table fdref \
    --out "$ROOT/chladni_hee/notes/hadamard_fdref_out"
