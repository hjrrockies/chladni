#!/bin/bash
#SBATCH --job-name=chladni_hadamard_hard_shapes
#SBATCH --array=0-49
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --partition=normal_q
#SBATCH --account=siallocation
#SBATCH --output=hadamard_hard_shapes_%A_%a.out

# notes/hadamard_hard_shapes.py on ARC: old chladni (e1ebaf8, what thesis_ratio_v1/v2 ran) against
# boundary-integrals-accuracy (chladni PR #21), on the final shapes of the hardest ratio runs.
# Tasks 0-9 are the FD references (the long ones), 10-49 the Hadamard probes; N_TASKS in the
# script must equal the array size.
#
# Once, from ~/projects/thesis (the douse runbook's layout, douse/docs/arc.md):
#   git -C chladni fetch origin
#   git -C chladni worktree add ../chladni_hs_new origin/boundary-integrals-accuracy
#   git -C chladni worktree add ../chladni_hs_old e1ebaf8
# Submit FROM the notes directory of the new checkout (the logs land there):
#   cd ~/projects/thesis/chladni_hs_new/notes && sbatch --mail-user=$DOUSE_SLURM_MAIL --mail-type=END,FAIL hadamard_hard_shapes.sbatch
# Then:
#   ~/projects/thesis/douse/.venv/bin/python hadamard_hard_shapes.py analyze hadamard_hard_shapes_out
#
# douse's venv is used for its dependencies only: each task puts the checkout it wants first on
# PYTHONPATH and refuses if chladni was imported from anywhere else. Neither checkout changes
# dependencies relative to the other.
set -euo pipefail
ROOT=${THESIS_ROOT:-$HOME/projects/thesis}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$ROOT/douse/.venv/bin/python" "$ROOT/chladni_hs_new/notes/hadamard_hard_shapes.py" task "$SLURM_ARRAY_TASK_ID" \
    --old "$ROOT/chladni_hs_old/src" --new "$ROOT/chladni_hs_new/src" \
    --out "$ROOT/chladni_hs_new/notes/hadamard_hard_shapes_out"
