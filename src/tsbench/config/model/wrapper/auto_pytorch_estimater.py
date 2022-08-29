from typing import Dict, List, Optional, Tuple

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset

from gluonts.dataset.common import Dataset
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor
import numpy as np
import os
import copy
from pathlib import Path
import shutil

from .auto_pytorch_predictor import AutoPytorchPredictor
try:
    from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
    from autoPyTorch.datasets.resampling_strategy import HoldoutValTypes
except ImportError:
    TimeSeriesForecastingTask = None

AUTOPYTORCH_IS_INSTALLED = TimeSeriesForecastingTask is not None

USAGE_MESSAGE = """
Cannot import `autopytorch`.

The `AutoPytorchEstimator` is a thin wrapper for calling the `AutoPytorch` package.
"""

FREQ_MAP = {
    "M": "1M",
    "Y": "1Y",
    "Q": "1Q",
    "D": "1D",
    "W": "1W",
    "H": "1H",
    "1H": "1H",
    "min": "1min",
    "10min": "10min",
    "0.5H": "30min"
}

class AutoPytorchEstimator(Estimator):
    """
    Wrapper around `Autogluon <https://github.com/awslabs/autogluon>`_.

    The `AutoGluonPredictor` is a thin wrapper for calling the `Autogluon`

    Parameters
    ----------
    freq
        Time frequency of the data, e.g. '1H'
    prediction_length
        Number of time points to predict
    run_time
        The time limit parameter for autogluon
    eval_metric
        The metric score in leaderboard results
    presets
        The preset parameter used in autogluon
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        run_time: int,
        optimize_metric: str,
        seed: int,
    ) -> None:
        super().__init__(freq=freq, prediction_length=prediction_length)

        if not AUTOPYTORCH_IS_INSTALLED:
            raise ImportError(USAGE_MESSAGE)

        self.freq = freq
        self.prediction_length = prediction_length
        self.run_time = run_time
        self.optimize_metric = optimize_metric
        self.seed = seed

        resampling_strategy = HoldoutValTypes.time_series_hold_out_validation
        resampling_strategy_args = None
        self.autopytorchts = TimeSeriesForecastingTask(
            seed=self.seed,
            ensemble_size=20,
            resampling_strategy=resampling_strategy,
            resampling_strategy_args=resampling_strategy_args,
        )
        self.autopytorchts.set_pipeline_config(device="cpu",
                                torch_num_threads=8,
                                early_stopping=20)
        
    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        **kwargs,
    ) -> AutoPytorchPredictor:
        val_target, val_start = self._data_process(validation_data)

        budget_kwargs = {"budget_type": "epochs",
                        "max_budget": 50,
                        "min_budget": 5}

        self.autopytorchts.search(
            X_train=None,
            optimize_metric=self.optimize_metric,
            y_train=list(copy.deepcopy(val_target)),
            n_prediction_steps=self.prediction_length,
            **budget_kwargs,
            freq=FREQ_MAP[self.freq],
            start_times=val_start,
            memory_limit=32 * 1024,
            normalize_y=False,
            total_walltime_limit=self.run_time,
            min_num_test_instances=1000,
        )

        # refit use train and val
        # TODO this version of Auto-PyTorch has a bug in reift, it will fixed in future version
        # remove refit might slightly weaken the final performance
        refit_dataset = self.autopytorchts.dataset.create_refit_set()
        try:
            print("refit")
            self.autopytorchts.refit(refit_dataset, 0)
        except Exception as e:
            print(e)

        print("autopytorch runtime:", self.run_time)
        return AutoPytorchPredictor(self.autopytorchts, prediction_length=self.prediction_length, freq=self.freq)

    def _data_process(self, dataset: Dataset) -> Tuple[np.array, List]:
        target = np.array([item["target"] for item in dataset])
        start = [item["start"] for item in dataset]
        return target, start

    def create_predictor(
        self,
    ) -> Predictor:
        return AutoPytorchEstimator(self.autopytorchts, prediction_length=self.prediction_length, freq=self.freq)
