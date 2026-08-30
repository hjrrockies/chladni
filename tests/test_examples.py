"""Execute the example notebooks.

An example that breaks silently is worse than no example, so the tour is run end to end
against the current API. Marked slow -- it spins up a kernel and costs ~5 s -- and slow tests are deselected by
default here. Run it with `pytest -m slow`.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "examples").glob("*.ipynb"))


def test_there_are_notebooks_to_check():
    assert NOTEBOOKS, "examples/ has no notebooks"


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_executes_cleanly(path):
    """Every cell runs, and the notebook's own asserts hold.

    The tour asserts its quantitative claims inline -- the arclength invariant, the
    divergence theorem, the closed-form metrics -- so an API change that silently alters a
    number fails here rather than printing something wrong to a reader.
    """
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    )
    client.execute()  # raises CellExecutionError on any failure, asserts included

    errors = [o for c in nb.cells for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors, errors[:1]
