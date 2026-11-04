#!/bin/bash
#SBATCH --job-name=chladni_accuracy_studies
#SBATCH --array=0-243
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --partition=normal_q
#SBATCH --account=siallocation
#SBATCH --output=accuracy_studies_%A_%a.out

# notes/accuracy_studies.py on ARC: the Bessel-bound check on Linux's scipy, the before/after
# eigenvalue and Hadamard study on the douse reference shapes, and the starved-basis Gram study.
# The array size must equal `python accuracy_studies.py ntasks` (244).
#
# Three checkouts, from ~/projects/thesis (the douse runbook's layout, douse/docs/arc.md):
#   git -C chladni fetch origin
#   git -C chladni worktree add ../chladni_acc     origin/accuracy-honesty   # scripts; tip src
#   git -C chladni worktree add ../chladni_acc_old 5a02def   # before the series-branch fix
#   git -C chladni worktree add ../chladni_acc_new 9de388a   # with it, and nothing else
# Submit FROM the notes directory of the tip checkout (the logs land there):
#   cd ~/projects/thesis/chladni_acc/notes && sbatch --mail-user=$DOUSE_SLURM_MAIL --mail-type=END,FAIL accuracy_studies.sbatch
# Then:
#   ~/projects/thesis/douse/.venv/bin/python accuracy_studies.py analyze accuracy_studies_out
#
# douse's venv supplies the dependencies, but NOT mpmath (the Bessel tasks need it): install it once,
#   uv pip install --python ~/projects/thesis/douse/.venv/bin/python mpmath
# (a later `uv sync` removes it again). Each task puts the checkout it wants
# first on PYTHONPATH and refuses if chladni was imported from anywhere else.
set -euo pipefail
ROOT=${THESIS_ROOT:-$HOME/projects/thesis}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$ROOT/douse/.venv/bin/python" "$ROOT/chladni_acc/notes/accuracy_studies.py" task "$SLURM_ARRAY_TASK_ID" \
    --tip "$ROOT/chladni_acc/src" --old "$ROOT/chladni_acc_old/src" --new "$ROOT/chladni_acc_new/src" \
    --out "$ROOT/chladni_acc/notes/accuracy_studies_out"
