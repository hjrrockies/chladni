#!/bin/bash
#SBATCH --job-name=chladni_feasible_full
#SBATCH --array=0-843%200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --partition=normal_q
#SBATCH --account=siallocation
#SBATCH --output=feasible_set_full_%A_%a.out

# notes/feasible_set.py on ARC: the FULL sweep, every family at every parameter and prec, no knobs. Two arrays, from the pilot's timings (2026-11-08): the tasks the pilot could not finish in 24 h get 72 h and 32G (`feasible_set.py array slow`). Each task writes its JSON as each stage lands (`status: partial`), so a timed-out task keeps its base solve and report; resubmitting finishes partial tasks and skips done ones.
#
# Run on the locked tag, from its OWN checkout, once no jobs are running from it. From ~/projects/thesis:
#   git -C chladni fetch origin --tags
#   git -C chladni_fs checkout thesis-lock-1
# Submit FROM its notes directory (the logs land there):
#   cd ~/projects/thesis/chladni_fs/notes
#   PY=~/projects/thesis/douse/.venv/bin/python
#   sbatch --array=$($PY feasible_set.py array normal)%200 --mail-user=$DOUSE_SLURM_MAIL --mail-type=END,FAIL feasible_set_full.sbatch
#   sbatch --array=$($PY feasible_set.py array slow)%60 --mem=32G --time=72:00:00 --mail-user=$DOUSE_SLURM_MAIL --mail-type=END,FAIL feasible_set_full.sbatch
# Then:
#   $PY feasible_set.py analyze feasible_set_out
set -euo pipefail
ROOT=${THESIS_ROOT:-$HOME/projects/thesis}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# The code is the checkout this was submitted from (its notes/ directory), so a rerun can come
# from a second checkout while the first one's tasks are still running; the output directory
# stays in chladni_fs, so every task's JSON lands in one place.
CODE="$(cd "$SLURM_SUBMIT_DIR/.." && pwd)"
export PYTHONPATH="$CODE/src"

"$ROOT/douse/.venv/bin/python" "$CODE/notes/feasible_set.py" task "$SLURM_ARRAY_TASK_ID" \
     --out "$ROOT/chladni_fs/notes/feasible_set_out"
