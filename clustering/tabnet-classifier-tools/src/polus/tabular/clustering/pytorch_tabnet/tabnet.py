"""Pytorch TabNet tool."""

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import polus.tabular.clustering.pytorch_tabnet.utils as ut
import vaex
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.tab_model import TabNetRegressor

# Initialize the logger
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("POLUS_LOG", logging.INFO))


class PytorchTabnet(ut.TabnetParameters):
    """Train a Pytorch TabNet model and evaluate it on validation data.

    Args:
        file_path: Path to the tabular data used for training.
        target_var: Dataset feature with classification labels.
        classifier: Select TabNetClassifier,TabNetMultiTaskClassifier,TabNetRegressor.
        out_dir: Path to the output directory.
    """

    file_path: Path
    target_var: str
    classifier: ut.Classifier
    out_dir: Path

    @property
    def convert_vaex_dataframe(self) -> vaex.dataframe.DataFrame:
        """Vaex supports reading tabular data in .csv, .feather, and .arrow formats."""
        extensions = [".arrow", ".feather", ".parquet"]
        if self.file_path.name.endswith(".csv"):
            return vaex.read_csv(
                Path(self.file_path),
                convert=True,
                chunk_size=5_000_000,
            )
        if self.file_path.name.endswith(tuple(extensions)):
            return vaex.open(Path(self.file_path))
        return None

    @property
    def get_data(self) -> ut.MyTupleType:
        """Subsetting for train/validation,extracting categorical indices/dimensions."""
        data = self.convert_vaex_dataframe

        if not isinstance(data.shape, tuple) and all(el != 0 for el in data.shape):
            msg = "Vaex dataframe is empty"
            raise ValueError(msg)

        if self.target_var not in list(data.columns):
            msg = f"{self.target_var} does not exist!!"
            raise ValueError(msg)

        features = [
            feature for feature in data.get_column_names() if feature != self.target_var
        ]

        cat_idxs = []
        cat_dims = []
        for i, col in enumerate(list(data.columns)):
            unique_values = 200
            if data[col].dtype == "string" or len(data[col].unique()) < unique_values:
                l_enc = LabelEncoder()
                data[col] = data[col].fillna("fillna")
                data[col] = l_enc.fit_transform(data[col].values)
                if col != self.target_var:
                    cat_idxs.append(i)
                    cat_dims.append(len(l_enc.classes_))
            else:
                # Calculate the mean of the column, ignoring NA values
                column_mean = data[col].mean()
                # Replace NA values with the column mean
                data[col] = data[col].fillna(column_mean)

        if len(cat_idxs) == 0 and len(cat_dims) == 0:
            cat_idxs = []
            cat_dims = []
            logger.info("Categorical features are not dectected")

        features = [
            feature for feature in data.get_column_names() if feature != self.target_var
        ]

        x = np.array(data[features])
        if self.classifier.value in ["TabNetRegressor", "TabNetMultiTaskClassifier"]:
            y = data[self.target_var].to_numpy().reshape(-1, 1)
        else:
            y = data[self.target_var].to_numpy()

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.test_size,
            random_state=42,
            stratify=y,
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=self.test_size,
            random_state=42,
            stratify=y_train,
        )

        return (
            x_train,
            y_train,
            x_test,
            y_test,
            x_val,
            y_val,
            cat_idxs,
            cat_dims,
            features,
        )

    @staticmethod
    def parameters(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Segmenting input parameters for model and evaluation."""
        exclude_params = [
            "out_dir",
            "file_path",
            "classifier",
            "test_size",
            "target_var",
        ]

        evalparams = [
            "eval_metric",
            "max_epochs",
            "patience",
            "weights",
            "loss_fn",
            "batch_size",
            "virtual_batch_size",
            "num_workers",
            "drop_last",
            "warm_start",
            "compute_importance",
        ]

        params = {
            k: v for k, v in params.items() if k not in exclude_params if v is not None
        }

        model_params = {k: v for k, v in params.items() if k not in evalparams}
        eval_params = {k: v for k, v in params.items() if k in evalparams}

        return model_params, eval_params

    def fit_model(self, params: dict[str, Any]) -> None:
        """Train a PyTorch tabNet model."""
        (
            x_train,
            y_train,
            _,
            _,
            x_val,
            y_val,
            cat_idxs,
            cat_dims,
            features,
        ) = self.get_data

        model_params, eval_params = self.parameters(params)

        model_params["cat_idxs"] = cat_idxs
        model_params["cat_dims"] = cat_dims

        eval_metric = eval_params["eval_metric"]

        if self.classifier.value == "TabNetClassifier":
            model = TabNetClassifier(**model_params)
            if eval_metric not in ut.BINARY_EVAL_METRIC:
                msg = f"Invalid eval_metric: {eval_metric} for {self.classifier.value}"
                raise ValueError(msg)

        if self.classifier.value == "TabNetMultiTaskClassifier":
            model = TabNetMultiTaskClassifier(**model_params)
            if eval_metric not in ut.MULTICLASS_EVAL_METRIC:
                msg = f"Invalid eval_metric: {eval_metric} for {self.classifier.value}"
                raise ValueError(msg)

        if self.classifier.value == "TabNetRegressor":
            model = TabNetRegressor(**model_params)
            if eval_metric not in ut.REGRESSION_EVAL_METRIC:
                msg = f"Invalid eval_metric: {eval_metric} for {self.classifier.value}"
                raise ValueError(msg)

        # This illustrates the behaviour of the model's fit method using
        # Compressed Sparse Row matrices
        sparse_x_train = csr_matrix(x_train)
        sparse_x_val = csr_matrix(x_val)

        model.fit(
            X_train=sparse_x_train,
            y_train=y_train,
            eval_set=[(x_train, y_train), (sparse_x_val, y_val)],
            eval_name=["train", "valid"],
            eval_metric=[eval_params["eval_metric"]],
            max_epochs=eval_params["max_epochs"],
            patience=eval_params["patience"],
            weights=eval_params["weights"],
            batch_size=eval_params["batch_size"],
            virtual_batch_size=eval_params["virtual_batch_size"],
            num_workers=eval_params["num_workers"],
            drop_last=eval_params["drop_last"],
            warm_start=eval_params["warm_start"],
            compute_importance=eval_params["compute_importance"],
        )

        # save tabnet model
        model_name = f"tabnet_{Path(self.file_path.name).stem}"
        model_path = self.out_dir.joinpath(model_name)
        logger.info("Saving of trained model")
        model.save_model(model_path)

        imp_features = [round(i, 4) for i in model.feature_importances_]

        feature_importance_pairs = list(zip(features, imp_features))
        sorted_feature_importance_pairs = dict(
            sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True),
        )

        save_feat_path = self.out_dir.joinpath("feature_importances.json")
        with Path.open(save_feat_path, "w") as jf:
            logger.info("Save feature importances")
            json.dump(sorted_feature_importance_pairs, jf, indent=4)
