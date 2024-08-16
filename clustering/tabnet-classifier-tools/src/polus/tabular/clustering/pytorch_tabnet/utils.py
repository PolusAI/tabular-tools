"""Pytorch TabNet tool."""

import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union

import numpy as np
import torch
from pydantic import BaseModel
from pydantic import Field

POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".arrow")

BINARY_EVAL_METRIC = ["auc", "accuracy", "balanced_accuracy", "logloss"]
MULTICLASS_EVAL_METRIC = ["accuracy", "balanced_accuracy", "logloss"]
REGRESSION_EVAL_METRIC = ["mse", "mae", "rmse", "rmsle"]


MyTupleType = tuple[
    np.ndarray,
    np.array,
    np.ndarray,
    np.array,
    np.ndarray,
    np.array,
    list[int],
    list[int],
    list[str],
]


def generate_preview(
    path: Path,
) -> None:
    """Generate preview of the plugin outputs."""
    source_path = Path(__file__).parents[5].joinpath("examples")
    shutil.copytree(source_path, path, dirs_exist_ok=True)


class OptimizersFn(str, Enum):
    """Optimizers Function."""

    Adadelta = "Adadelta"
    Adagrad = "Adagrad"
    Adam = "Adam"
    AdamW = "AdamW"
    SparseAdam = "SparseAdam"
    Adamax = "Adamax"
    ASGD = "ASGD"
    LBFGS = "LBFGS"
    NAdam = "NAdam"
    RAdam = "RAdam"
    RMSprop = "RMSprop"
    Rprop = "Rprop"
    SGD = "SGD"
    Default = "Adam"


Map_OptimizersFn = {
    OptimizersFn.Adadelta: torch.optim.Adadelta,
    OptimizersFn.Adagrad: torch.optim.Adagrad,
    OptimizersFn.Adam: torch.optim.Adam,
    OptimizersFn.AdamW: torch.optim.AdamW,
    OptimizersFn.SparseAdam: torch.optim.SparseAdam,
    OptimizersFn.Adamax: torch.optim.Adamax,
    OptimizersFn.ASGD: torch.optim.ASGD,
    OptimizersFn.LBFGS: torch.optim.LBFGS,
    OptimizersFn.ASGD: torch.optim.ASGD,
    OptimizersFn.LBFGS: torch.optim.LBFGS,
    OptimizersFn.NAdam: torch.optim.NAdam,
    OptimizersFn.RAdam: torch.optim.RAdam,
    OptimizersFn.RMSprop: torch.optim.RMSprop,
    OptimizersFn.Rprop: torch.optim.Rprop,
    OptimizersFn.SGD: torch.optim.SGD,
    OptimizersFn.Default: torch.optim.Adam,
}


class SchedulerFn(str, Enum):
    """Scheduler Function."""

    LambdaLR = "LambdaLR"
    MultiplicativeLR = "MultiplicativeLR"
    StepLR = "StepLR"
    MultiStepLR = "MultiStepLR"
    ConstantLR = "ConstantLR"
    LinearLR = "LinearLR"
    ExponentialLR = "ExponentialLR"
    PolynomialLR = "PolynomialLR"
    CosineAnnealingLR = "CosineAnnealingLR"
    ChainedScheduler = "ChainedScheduler"
    SequentialLR = "SequentialLR"
    CyclicLR = "CyclicLR"
    OneCycleLR = "OneCycleLR"
    CosineAnnealingWarmRestarts = "CosineAnnealingWarmRestarts"
    Default = "StepLR"


Map_SchedulerFn = {
    SchedulerFn.LambdaLR: torch.optim.lr_scheduler.LambdaLR,
    SchedulerFn.MultiplicativeLR: torch.optim.lr_scheduler.MultiplicativeLR,
    SchedulerFn.StepLR: torch.optim.lr_scheduler.StepLR,
    SchedulerFn.MultiStepLR: torch.optim.lr_scheduler.MultiStepLR,
    SchedulerFn.ConstantLR: torch.optim.lr_scheduler.ConstantLR,
    SchedulerFn.LinearLR: torch.optim.lr_scheduler.LinearLR,
    SchedulerFn.ExponentialLR: torch.optim.lr_scheduler.ExponentialLR,
    SchedulerFn.PolynomialLR: torch.optim.lr_scheduler.PolynomialLR,
    SchedulerFn.CosineAnnealingLR: torch.optim.lr_scheduler.CosineAnnealingLR,
    SchedulerFn.ChainedScheduler: torch.optim.lr_scheduler.ChainedScheduler,
    SchedulerFn.SequentialLR: torch.optim.lr_scheduler.SequentialLR,
    SchedulerFn.CyclicLR: torch.optim.lr_scheduler.CyclicLR,
    SchedulerFn.OneCycleLR: torch.optim.lr_scheduler.OneCycleLR,
    SchedulerFn.CosineAnnealingWarmRestarts: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,  # noqa : E501
    SchedulerFn.Default: torch.optim.lr_scheduler.StepLR,
}


class Evalmetric(str, Enum):
    """Evaluation Metric."""

    AUC = "auc"
    ACCURACY = "accuracy"
    BALANCEDACCURACY = "balanced_accuracy"
    LOGLOSS = "logloss"
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    RMSLE = "rmsle"
    DEFAULT = "auc"


class MaskType(str, Enum):
    """Masking Function."""

    SPARSEMAX = "sparsemax"
    ENTMAX = "entmax"
    DEFAULT = "entmax"


class DeviceName(str, Enum):
    """Platform Name."""

    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"
    DEFAULT = "auto"


class Classifier(str, Enum):
    """Pytorch TabNet Classifier."""

    TabNetClassifier = "TabNetClassifier"
    TabNetRegressor = "TabNetRegressor"
    TabNetMultiTaskClassifier = "TabNetMultiTaskClassifier"
    DEFAULT = "TabNetClassifier"


class LossFunctions(str, Enum):
    """Loss Functions."""

    L1Loss = "L1Loss"
    NLLLoss = "NLLLoss"
    NLLLoss2d = "NLLLoss2d"
    PoissonNLLLoss = "PoissonNLLLoss"
    GaussianNLLLoss = "GaussianNLLLoss"
    KLDivLoss = "KLDivLoss"
    MSELoss = "MSELoss"
    BCELoss = "BCELoss"
    BCEWithLogitsLoss = "BCEWithLogitsLoss"
    HingeEmbeddingLoss = "HingeEmbeddingLoss"
    SmoothL1Loss = "SmoothL1Loss"
    HuberLoss = "HuberLoss"
    SoftMarginLoss = "SoftMarginLoss"
    CrossEntropyLoss = "CrossEntropyLoss"
    MultiLabelSoftMarginLoss = "MultiLabelSoftMarginLoss"
    CosineEmbeddingLoss = "CosineEmbeddingLoss"
    MarginRankingLoss = "MarginRankingLoss"
    MultiMarginLoss = "MultiMarginLoss"
    TripletMarginLoss = "TripletMarginLoss"
    TripletMarginWithDistanceLoss = "TripletMarginWithDistanceLoss"
    CTCLoss = "CTCLoss"
    DEFAULT = "MSELoss"


class TabnetParameters(BaseModel):
    """Parameters for Pytorch TabNet model."""

    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    n_d: int = Field(default=8, ge=8, le=64)
    n_a: int = Field(default=8, ge=8, le=64)
    n_steps: int = Field(default=3, ge=3, le=10)
    gamma: float = Field(default=1.3, ge=1.0, le=2.0)
    cat_emb_dim: int = Field(default=1)
    n_independent: int = Field(default=2, ge=1, le=5)
    n_shared: int = Field(default=2, ge=1, le=5)
    epsilon: float = Field(default=1e-15)
    seed: int = Field(default=0)
    momentum: float = Field(default=0.02, ge=0.01, le=0.4)
    clip_value: Union[float, None] = Field(default=None)
    lambda_sparse: float = Field(default=1e-3)
    optimizer_fn: Any = Field(default=torch.optim.Adam)
    optimizer_params: dict = Field(default={"lr": 0.02})
    scheduler_fn: Any = Field(default=torch.optim.lr_scheduler.StepLR)
    scheduler_params: dict = Field(default={"step_size": 10, "gamma": 0.95})
    device_name: str = Field(default="auto")
    mask_type: str = Field(default="entmax")
    grouped_features: Optional[Union[list[int], None]] = Field(default=None)
    n_shared_decoder: int = Field(default=1)
    n_indep_decoder: int = Field(default=1)
    eval_metric: str = Field(default="auc")
    max_epochs: int = Field(default=200)
    patience: int = Field(default=10)
    weights: int = Field(default=0)
    loss_fn: str = Field(default="MSELoss")
    batch_size: int = Field(default=1024)
    virtual_batch_size: int = Field(default=128)
    num_workers: int = Field(default=0)
    drop_last: bool = Field(default=False)
    warm_start: bool = Field(default=False)
    compute_importance: bool = Field(default=True)
