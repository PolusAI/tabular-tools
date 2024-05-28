"""Test Fixtures."""

import shutil
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytest

EXT = None


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to pytest."""
    parser.addoption(
        "--slow",
        action="store_true",
        dest="slow",
        default=False,
        help="run slow tests",
    )


def clean_directories() -> None:
    """Remove all temporary directories."""
    for d in Path(".").cwd().iterdir():
        if d.is_dir() and d.name.startswith("tmp"):
            shutil.rmtree(d)


@pytest.fixture()
def input_directory() -> Union[str, Path]:
    """Create input directory."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture()
def output_directory() -> Union[str, Path]:
    """Create output directory."""
    return Path(tempfile.mkdtemp(dir=Path.cwd()))


@pytest.fixture()
def create_dataset() -> Union[str, Path]:
    """Create output directory."""
    size = 5000

    inp_dir = Path(tempfile.mkdtemp(dir=Path.cwd()))

    rng = np.random.default_rng()

    workclass = [
        "Private",
        "Local-gov",
        "Self-emp-not-inc",
        "Federal-gov",
        "State-gov",
        "Self-emp-inc",
        "Without-pay",
        "Never-worked",
    ]

    education = [
        "11th",
        "HS-grad",
        "Assoc-acdm",
        "Some-college",
        "10th",
        "Prof-school",
        "7th-8th",
        "Bachelors",
        "Masters",
        "Doctorate",
        "5th-6th",
        "Assoc-voc",
        "9th",
        "12th",
        "1st-4th",
        "Preschool",
    ]

    marital_status = [
        "Never-married",
        "Married-civ-spouse",
        "Widowed",
        "Divorced",
        "Separated",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ]

    occupation = [
        "Machine-op-inspct",
        "Farming-fishing",
        "Protective-serv",
        "?",
        "Other-service",
        "Prof-specialty",
        "Craft-repair",
        "Adm-clerical",
        "Exec-managerial",
        "Tech-support",
        "Sales",
        "Priv-house-serv",
        "Transport-moving",
        "Handlers-cleaners",
        "Armed-Forces",
    ]

    relationship = [
        "Own-child",
        "Husband",
        "Not-in-family",
        "Unmarried",
        "Wife",
        "Other-relative",
    ]

    race = ["Black", "White", "Asian-Pac-Islander", "Other", "Amer-Indian-Eskimo"]

    gender = ["Male", "Female"]

    income = ["<=50K", ">50K"]

    countries = [
        "United-States",
        "Peru",
        "Guatemala",
        "Mexico",
        "Dominican-Republic",
        "Ireland",
        "Germany",
        "Philippines",
        "Thailand",
        "Haiti",
        "El-Salvador",
        "Puerto-Rico",
        "Vietnam",
        "South",
        "Columbia",
        "Japan",
        "India",
        "Cambodia",
        "Poland",
        "Laos",
        "England",
        "Cuba",
        "Taiwan",
        "Italy",
        "Canada",
        "Portugal",
        "China",
        "Nicaragua",
        "Honduras",
        "Iran",
        "Scotland",
        "Jamaica",
        "Ecuador",
        "Yugoslavia",
        "Hungary",
        "Hong",
        "Greece",
        "Trinadad&Tobago",
        "Outlying-US(Guam-USVI-etc)",
        "France",
        "Holand-Netherlands",
    ]

    diction_1 = {
        "age": rng.integers(low=15, high=80, size=size),
        "workclass": rng.choice(workclass, size),
        "fnlwgt": rng.integers(low=12285, high=1490400, size=size),
        "education": rng.choice(education, size),
        "educational-num": rng.integers(low=1, high=16, size=size),
        "marital-status": rng.choice(marital_status, size),
        "occupation": rng.choice(occupation, size),
        "relationship": rng.choice(relationship, size),
        "race": rng.choice(race, size),
        "gender": rng.choice(gender, size),
        "capital-gain": rng.integers(low=0, high=99999, size=size),
        "capital-loss": rng.integers(low=0, high=4356, size=size),
        "hours-per-week": rng.integers(low=1, high=99, size=size),
        "native-country": rng.choice(countries, size),
        "income": rng.choice(income, size),
    }

    data = pd.DataFrame(diction_1)

    data.to_csv(Path(inp_dir, "adult.csv"), index=False)
    data.to_feather(Path(inp_dir, "adult.arrow"))
    data.to_parquet(Path(inp_dir, "adult.parquet"))

    return inp_dir


@pytest.fixture(
    params=[
        (0.3, "Adam", "StepLR", "accuracy", "L1Loss", "TabNetMultiTaskClassifier"),
        (0.2, "Adadelta", "StepLR", "mse", "GaussianNLLLoss", "TabNetRegressor"),
        (0.2, "Adagrad", "StepLR", "logloss", "MSELoss", "TabNetClassifier"),
        (0.2, "RAdam", "StepLR", "auc", "CrossEntropyLoss", "TabNetClassifier"),
    ],
)
def get_params(request: pytest.FixtureRequest) -> pytest.FixtureRequest:
    """To get the parameter of the fixture."""
    return request.param
