"""Pytorch TabNet tool."""

import filepattern as fp
import pytest
import torch
import polus.tabular.clustering.pytorch_tabnet.tabnet as tb
from .conftest import clean_directories
from pathlib import Path
from typing import Union
import numpy as np


# @pytest.mark.skipif("not config.getoption('slow')")
def test_convert_vaex_dataframe(
    output_directory: Path,
    create_dataset: Union[str, Path],
    get_params: pytest.FixtureRequest,
) -> None:
    """Testing reading vaex dataframe."""

    inp_dir = create_dataset
    test_size, optimizer_fn, scheduler_fn, eval_metric, loss_fn, classifier = get_params

    params = {
        "test_size": test_size,
        "n_d": 8,
        "n_a": 8,
        "seed": 0,
        "optimizer_fn": optimizer_fn,
        "optimizer_params": {"lr": 0.001},
        "scheduler_fn": scheduler_fn,
        "device_name": "cpu",
        "eval_metric": eval_metric,
        "max_epochs": 10,
        "loss_fn": loss_fn,
    }

    patterns = [".*.csv", ".*.arrow", ".*.parquet"]

    for pat in patterns:
        fps = fp.FilePattern(inp_dir, pat)

        for f in fps:
            model = tb.PytorchTabnet(
                **params,
                file_path=f[1][0],
                target_var="income",
                classifier=classifier,
                out_dir=output_directory,
            )
            df = model.convert_vaex_dataframe
            assert df.shape == (5000, 15)
            assert df is not None

    clean_directories()


def test_get_data(
    output_directory: Path,
    create_dataset: Union[str, Path],
    get_params: pytest.FixtureRequest,
) -> None:
    """Testing getting data."""

    inp_dir = create_dataset
    test_size, optimizer_fn, scheduler_fn, eval_metric, loss_fn, classifier = get_params

    params = {
        "test_size": test_size,
        "n_d": 8,
        "n_a": 8,
        "seed": 0,
        "optimizer_fn": optimizer_fn,
        "optimizer_params": {"lr": 0.001},
        "scheduler_fn": scheduler_fn,
        "device_name": "cpu",
        "eval_metric": eval_metric,
        "max_epochs": 10,
        "loss_fn": loss_fn,
    }
    fps = fp.FilePattern(inp_dir, ".*.csv")
    file = [f[1][0] for f in fps()][0]

    model = tb.PytorchTabnet(
        **params,
        file_path=file,
        target_var="income",
        classifier=classifier,
        out_dir=output_directory,
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
        cat_idxs,
        cat_dims,
        features,
    ) = model.get_data

    assert all(
        isinstance(arr, np.ndarray)
        for arr in [X_train, y_train, X_test, y_test, X_val, y_val]
    )
    assert all(isinstance(i, list) for i in [cat_idxs, cat_dims, features])


def test_fit_model(
    output_directory: Path,
    create_dataset: Union[str, Path],
    get_params: pytest.FixtureRequest,
) -> None:
    """Testing fitting model."""

    inp_dir = create_dataset
    test_size, _, _, eval_metric, loss_fn, classifier = get_params

    params = {
        "test_size": test_size,
        "n_d": 8,
        "n_a": 8,
        "seed": 0,
        "optimizer_fn": torch.optim.Adam,
        "scheduler_fn": torch.optim.lr_scheduler.StepLR,
        "device_name": "cpu",
        "eval_metric": eval_metric,
        "max_epochs": 10,
        "loss_fn": loss_fn,
    }
    fps = fp.FilePattern(inp_dir, ".*.csv")
    file = [f[1][0] for f in fps()][0]

    model = tb.PytorchTabnet(
        **params,
        file_path=file,
        target_var="income",
        classifier=classifier,
        out_dir=output_directory,
    )
    mod_params = dict(model)
    model.fit_model(params=mod_params)

    files = [
        f for f in Path(output_directory).iterdir() if f.suffix in [".zip", ".json"]
    ]

    assert len(files) != 0
    clean_directories()
