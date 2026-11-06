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

# notes/feasible_set.py on ARC (2026-11-06): the FULL sweep, every family at every parameter and prec, no knobs. Array size = `python feasible_set.py ntasks` (844); at most 200 run at once. Submit only after the pilot has been read: its timings may change --time and --mem, or the families kept.
#
# Its OWN checkout. From ~/projects/thesis:
#   git -C chladni fetch origin
#   git -C chladni worktree add ../chladni_fs origin/solve-report
# (later: git -C chladni_fs pull --ff-only  -- not while jobs from it are running)
# Submit FROM its notes directory (the logs land there):
#   cd ~/projects/thesis/chladni_fs/notes && sbatch --mail-user=$DOUSE_SLURM_MAIL --mail-type=END,FAIL feasible_set_full.sbatch
# Then:
#   ~/projects/thesis/douse/.venv/bin/python feasible_set.py analyze feasible_set_out
set -euo pipefail
ROOT=${THESIS_ROOT:-$HOME/projects/thesis}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$ROOT/chladni_fs/src"

"$ROOT/douse/.venv/bin/python" "$ROOT/chladni_fs/notes/feasible_set.py" task "$SLURM_ARRAY_TASK_ID" \
     --out "$ROOT/chladni_fs/notes/feasible_set_out"
