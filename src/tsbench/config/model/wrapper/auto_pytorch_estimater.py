from typing import Dict, List, Optional, Tuple

from mxnet.gluon import HybridBlock

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset

from gluonts.dataset.common import Dataset
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor
from gluonts.transform import Transformation
import numpy as np
import time
import os
import copy
import shutil
from pathlib import Path

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
    "1min": "1min",
    "10min": "10min",
    "30min": "30min"
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
        budget_type: str,
        run_time: int,
        eval_metric: str,
        presets: Optional[str],
    ) -> None:
        super().__init__(freq=freq, prediction_length=prediction_length)

        if not AUTOPYTORCH_IS_INSTALLED:
            raise ImportError(USAGE_MESSAGE)

        self.freq = freq
        self.prediction_length = prediction_length
        self.budget_type = budget_type
        self.run_time = run_time

        working_dir = os.getenv("SM_MODEL_DIR") or Path.home() / "models"
        path = Path(working_dir) / 'APT_run'
        now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
        path_log = str(path / "m3_monthly" / str(now) / budget_type / f'{10}' / "log")
        path_pred = str(path / "m3_monthly" / str(now) / budget_type / f'{10}' / "output")
        

        resampling_strategy = HoldoutValTypes.time_series_hold_out_validation
        resampling_strategy_args = None
        # Remove intermediate files
        try:
            shutil.rmtree(path_log)
            shutil.rmtree(path_pred)
            Path(path_log).mkdir(parents=True, exist_ok=True)
            Path(path_pred).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
        self.autopytorchts = TimeSeriesForecastingTask(
            seed=10,
            ensemble_size=20,
            resampling_strategy=resampling_strategy,
            resampling_strategy_args=resampling_strategy_args,
            temporary_directory=path_log,
            output_directory=path_pred,
        )
        self.autopytorchts.set_pipeline_config(device="cuda",
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
        # train_target, train_start = self._data_process(training_data)
        val_target, val_start = self._data_process(validation_data)

        if self.budget_type == "random_search":
            budget_kwargs = {'budget_type': 'random_search',
                            'max_budget': None,
                            'min_budget': None}

        elif self.budget_type != 'full_budget':
            from autoPyTorch.constants_forecasting import FORECASTING_BUDGET_TYPE
            if self.budget_type not in FORECASTING_BUDGET_TYPE and self.budget_type != 'epochs':
                raise NotImplementedError('Unknown Budget Type!')
            budget_kwargs = {'budget_type': self.budget_type,
                            'max_budget': 50 if self.budget_type == 'epochs' else 1.0,
                            'min_budget': 5 if self.budget_type == 'epochs' else 0.1}
        else:
            budget_kwargs = {'budget_type': 'epochs',
                            'max_budget': 50,
                            'min_budget': 50}

        self.autopytorchts.search(
            X_train=None,
            # TODO why copy
            y_train=copy.deepcopy(val_target),
            optimize_metric='mean_MASE_forecasting',
            n_prediction_steps=self.prediction_length,
            **budget_kwargs,
            freq=FREQ_MAP[self.freq],
            start_times_train=val_start,
            memory_limit=32 * 1024,
            normalize_y=False,
            total_walltime_limit=self.run_time,
            min_num_test_instances=1000,
        )

        # refit_dataset = self.autopytorchts.dataset.create_refit_set()
        # try:
        #     self.autopytorchts.refit(refit_dataset, 0)
        # except Exception as e:
        #     print(e)

        print("autopytorch runtime:", self.run_time)
        return AutoPytorchPredictor(self.autopytorchts, prediction_length=self.prediction_length, freq=self.freq)

    def _data_process(self, dataset: Dataset) -> Tuple[np.array, List]:
        target = np.array([item["target"] for item in dataset])
        start = [item["start"] for item in dataset]
        return target, start

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        return AutoPytorchEstimator(self.autopytorchts, prediction_length=self.prediction_length, freq=self.freq)
