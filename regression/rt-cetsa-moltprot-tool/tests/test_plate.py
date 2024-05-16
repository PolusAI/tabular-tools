import os
import pathlib

from polus.tabular.regression.rt_cetsa_moltprot.__main__ import main

INP_DATA_DIR = os.environ.get(
    "TEST_DATA_DIR",
    "/home/nishaq/Documents/axle/data/Data for Nick/20210318 LDHA compound plates/20210318 LDHA compound plate 1 6K cells",
)


def get_inp_dir():
    inp_dir = pathlib.Path(INP_DATA_DIR).resolve()
    assert inp_dir.exists(), f"Input directory does not exist: {inp_dir}"
    inp_dir = inp_dir.parent / "out-plate-1"
    assert inp_dir.exists(), f"Input directory does not exist: {inp_dir}"
    return inp_dir


def test_tool():
    inp_dir = get_inp_dir()

    main(
        inp_dir=inp_dir,
        pattern="plate.csv",
        preview=False,
        out_dir=inp_dir,
        baseline_fit=None,
        baseline_bounds=None,
        dCp=None,
        onset_threshold=None,
        savgol=None,
        trim_max=None,
        trim_min=None,
    )
