"""Pytorch TabNet tool."""

import logging
import os
from pathlib import Path
from typing import Optional
from typing import Union

import filepattern as fp
import polus.tabular.clustering.pytorch_tabnet.tabnet as pt
import polus.tabular.clustering.pytorch_tabnet.utils as ut
import typer

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("polus.tabular.clustering.pytorch_tabnet")
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))
POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".arrow")


app = typer.Typer()


@app.command()
def main(  # noqa: PLR0913 PLR0915
    inp_dir: Path = typer.Option(
        ...,
        "--inpDir",
        help="Input tabular data",
    ),
    file_pattern: str = typer.Option(
        ".+",
        "--filePattern",
        help="pattern to parse tabular files",
    ),
    test_size: float = typer.Option(
        0.2,
        "--testSize",
        help="Proportion of the dataset to include in the test set",
        min=0.1,
        max=0.4,
    ),
    n_d: Optional[Union[int, None]] = typer.Option(
        8,
        "--nD",
        help="Width of the decision prediction layer",
        min=8,
        max=64,
    ),
    n_a: Optional[Union[int, None]] = typer.Option(
        8,
        "--nA",
        help="Width of the attention embedding for each mask",
        min=8,
        max=64,
    ),
    n_steps: Optional[Union[int, None]] = typer.Option(
        3,
        "--nSteps",
        help="Number of steps in the architecture",
        min=3,
        max=10,
    ),
    gamma: Optional[Union[float, None]] = typer.Option(
        1.3,
        "--gamma",
        help="Coefficient for feature reuse in the masks",
        min=1.0,
        max=2.0,
    ),
    cat_emb_dim: Optional[Union[int, None]] = typer.Option(
        1,
        "--catEmbDim",
        help="List of embedding sizes for each categorical feature",
    ),
    n_independent: Optional[Union[int, None]] = typer.Option(
        2,
        "--nIndependent",
        help="Number of independent Gated Linear Unit layers at each step",
        min=1,
        max=5,
    ),
    n_shared: Optional[Union[int, None]] = typer.Option(
        2,
        "--nShared",
        help="Number of shared Gated Linear Unit layers at each step",
        min=1,
        max=5,
    ),
    epsilon: Optional[Union[float, None]] = typer.Option(
        1e-15,
        "--epsilon",
        help="Constant value",
    ),
    seed: Optional[Union[int, None]] = typer.Option(
        0,
        "--seed",
        help="Random seed for reproducibility",
    ),
    momentum: Optional[Union[float, None]] = typer.Option(
        0.02,
        "--momentum",
        help="Momentum for batch normalization",
        min=0.01,
        max=0.4,
    ),
    clip_value: Optional[Union[float, None]] = typer.Option(
        None,
        "--clipValue",
        help="Clipping of the gradient value",
    ),
    lambda_sparse: Optional[Union[float, None]] = typer.Option(
        1e-3,
        "--lambdaSparse",
        help="Extra sparsity loss coefficient",
    ),
    optimizer_fn: ut.OptimizersFn = typer.Option(
        ut.OptimizersFn.Default,
        "--optimizerFn",
        help="Pytorch optimizer function",
    ),
    lr: Optional[Union[float, None]] = typer.Option(
        0.02,
        "--lr",
        help="learning rate for the optimizer",
    ),
    scheduler_fn: ut.SchedulerFn = typer.Option(
        ut.SchedulerFn.Default,
        "--schedulerFn",
        help="Parameters used initialize the optimizer.",
    ),
    step_size: int = typer.Option(
        10,
        "--stepSize",
        help="Parameter to apply to the scheduler_fn.",
    ),
    device_name: ut.DeviceName = typer.Option(
        ut.DeviceName.DEFAULT,
        "--deviceName",
        help="Device used for training",
    ),
    mask_type: ut.MaskType = typer.Option(
        ut.MaskType.DEFAULT,
        "--maskType",
        help="A masking function for feature selection",
    ),
    grouped_features: Optional[Union[list[int], None]] = typer.Option(
        None,
        "--groupedFeatures",
        help="Allow the model to share attention across features within the same group",
    ),
    n_shared_decoder: Optional[Union[int, None]] = typer.Option(
        1,
        "--nSharedDecoder",
        help="Number of shared GLU block in decoder",
    ),
    n_indep_decoder: Optional[Union[int, None]] = typer.Option(
        1,
        "--nIndepDecoder",
        help="Number of independent GLU block in decoder",
    ),
    eval_metric: ut.Evalmetric = typer.Option(
        ut.Evalmetric.DEFAULT,
        "--evalMetric",
        help="Metrics utilized for early stopping evaluation",
    ),
    max_epochs: Optional[Union[int, None]] = typer.Option(
        200,
        "--maxEpochs",
        help="Maximum number of epochs for training",
    ),
    patience: Optional[Union[int, None]] = typer.Option(
        10,
        "--patience",
        help="Consecutive epochs without improvement before early stopping",
    ),
    weights: Optional[Union[int, None]] = typer.Option(
        0,
        "--weights",
        help="Sampling parameter only for TabNetClassifier",
    ),
    loss_fn: ut.LossFunctions = typer.Option(
        ut.LossFunctions.DEFAULT,
        "--lossFn",
        help="Loss function",
    ),
    batch_size: Optional[Union[int, None]] = typer.Option(
        1024,
        "--batchSize",
        help="Batch size",
    ),
    virtual_batch_size: Optional[Union[int, None]] = typer.Option(
        128,
        "--virtualBatchSize",
        help="Size of mini-batches for Ghost Batch Normalization.",
    ),
    num_workers: Optional[Union[int, None]] = typer.Option(
        0,
        "--numWorkers",
        help="Number or workers used in torch.utils.data.Dataloader",
    ),
    drop_last: bool = typer.Option(
        False,
        "--dropLast",
        help="Option to drop incomplete last batch during training",
    ),
    warm_start: bool = typer.Option(
        False,
        "--warmStart",
        help="For scikit-learn compatibility, enabling fitting the same model twice",
    ),
    compute_importance: bool = typer.Option(
        True,
        "--computeImportance",
        help="Compute feature importance?",
    ),
    target_var: str = typer.Option(
        ...,
        "--targetVar",
        help="Target feature for classification",
    ),
    classifier: ut.Classifier = typer.Option(
        ut.Classifier.DEFAULT,
        "--classifier",
        help="Tabnet Classifier",
    ),
    out_dir: Path = typer.Option(..., "--outDir", help="Output collection"),
    preview: Optional[bool] = typer.Option(
        False,
        "--preview",
        help="Output a JSON preview of outputs produced by this plugin",
    ),
) -> None:
    """Tool for training tabular data using PyTorch TabNet."""
    logger.info(f"--inpDir = {inp_dir}")
    logger.info(f"--filePattern = {file_pattern}")
    logger.info(f"--testSize = {test_size}")
    logger.info(f"--nD = {n_d}")
    logger.info(f"--nA = {n_a}")
    logger.info(f"--nSteps = {n_steps}")
    logger.info(f"--gamma = {gamma}")
    logger.info(f"--catEmbDim = {cat_emb_dim}")
    logger.info(f"--nIndependent = {n_independent}")
    logger.info(f"--nShared = {n_shared}")
    logger.info(f"--epsilon = {epsilon}")
    logger.info(f"--seed = {seed}")
    logger.info(f"--momentum = {momentum}")
    logger.info(f"--clipValue = {clip_value}")
    logger.info(f"--lambdaSparse = {lambda_sparse}")
    logger.info(f"--optimizerFn = {optimizer_fn}")
    logger.info(f"--lr = {lr}")
    logger.info(f"--schedulerFn = {scheduler_fn}")
    logger.info(f"--stepSize = {step_size}")
    logger.info(f"--deviceName = {device_name}")
    logger.info(f"--maskType = {mask_type}")
    logger.info(f"--groupedFeatures = {grouped_features}")
    logger.info(f"--nSharedDecoder = {n_shared_decoder}")
    logger.info(f"--nIndepDecode = {n_indep_decoder}")
    logger.info(f"--evalMetric = {eval_metric}")
    logger.info(f"--maxEpochs = {max_epochs}")
    logger.info(f"--patience = {patience}")
    logger.info(f"--weights = {weights}")
    logger.info(f"--lossFn = {loss_fn}")
    logger.info(f"--batch_size = {batch_size}")
    logger.info(f"--virtualBatchSize = {virtual_batch_size}")
    logger.info(f"--numWorkers = {num_workers}")
    logger.info(f"--dropLast = {drop_last}")
    logger.info(f"--warm_start = {warm_start}")
    logger.info(f"--computeImportance = {compute_importance}")
    logger.info(f"--targetVar = {target_var}")
    logger.info(f"--classifier = {classifier}")
    logger.info(f"--outDir = {out_dir}")

    if not Path(inp_dir).exists():
        msg = f"The input directory {Path(inp_dir).stem} does not exist."
        raise FileNotFoundError(msg)

    if not Path(out_dir).exists():
        Path(out_dir).mkdir(exist_ok=False, parents=True)
        msg = f"The output directory {out_dir} created."
        logger.info(msg)

    params = {
        "test_size": test_size,
        "n_d": n_d,
        "n_a": n_a,
        "n_steps": n_steps,
        "gamma": gamma,
        "cat_emb_dim": cat_emb_dim,
        "n_independent": n_independent,
        "n_shared": n_shared,
        "epsilon": epsilon,
        "seed": seed,
        "momentum": momentum,
        "clip_value": clip_value,
        "lambda_sparse": lambda_sparse,
        "optimizer_fn": ut.Map_OptimizersFn[optimizer_fn],
        "optimizer_params": {"lr": lr},
        "scheduler_fn": ut.Map_SchedulerFn[scheduler_fn],
        "scheduler_params": {"step_size": step_size, "gamma": 0.95},
        "device_name": device_name.value,
        "mask_type": mask_type.value,
        "grouped_features": grouped_features,
        "n_shared_decoder": n_shared_decoder,
        "n_indep_decoder": n_indep_decoder,
        "eval_metric": eval_metric.value,
        "max_epochs": max_epochs,
        "patience": patience,
        "weights": weights,
        "loss_fn": loss_fn.value,
        "batch_size": batch_size,
        "virtual_batch_size": virtual_batch_size,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "warm_start": warm_start,
        "compute_importance": compute_importance,
    }

    fps = fp.FilePattern(inp_dir, file_pattern)

    flist = [f[1][0] for f in fps()]

    if len(flist) == 0:
        msg = f"No files found with pattern: {file_pattern}."
        raise ValueError(msg)

    if preview:
        ut.generate_preview(out_dir)

    if not preview:
        for file in flist:
            model = pt.PytorchTabnet(
                **params,
                file_path=file,
                target_var=target_var,
                classifier=classifier,
                out_dir=out_dir,
            )
            mod_params = dict(model)

            model.fit_model(params=mod_params)


if __name__ == "__main__":
    app()
