from typing import Dict, Optional

from mxnet.gluon import HybridBlock

from gluonts.core.component import validated
from gluonts.dataset.common import Dataset

from gluonts.dataset.common import Dataset
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor
from gluonts.transform import Transformation

from .auto_gluon_predictor import AutoGluonPredictor
try:
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
except ImportError:
    TimeSeriesPredictor = None

AUTOGLUON_IS_INSTALLED = TimeSeriesPredictor is not None

USAGE_MESSAGE = """
Cannot import `autogluon`.

The `AutoGluonEstimator` is a thin wrapper for calling the `AutoGluon` package.
"""

class AutoGluonEstimator(Estimator):
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
        eval_metric: str,
        presets: Optional[str],
    ) -> None:
        super().__init__(freq=freq, prediction_length=prediction_length)

        if not AUTOGLUON_IS_INSTALLED:
            raise ImportError(USAGE_MESSAGE)

        self.freq = freq
        self.prediction_length = prediction_length
        self.autogluonts = TimeSeriesPredictor(prediction_length=prediction_length, eval_metric=eval_metric)
        self.presets = presets
        self.run_time = run_time

    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
    ) -> AutoGluonPredictor:

        train_dataframe = TimeSeriesDataFrame(training_data)
        valid_dataframe = TimeSeriesDataFrame(validation_data)

        tspredictor = self.autogluonts.fit(train_dataframe, tuning_data=valid_dataframe, presets=self.presets, time_limit=self.run_time)
        
        return AutoGluonPredictor(tspredictor, prediction_length=self.prediction_length, freq=self.freq)

        
    def train(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        **kwargs,
    ) -> AutoGluonPredictor:
        return self.train_model(
            training_data=training_data,
            validation_data=validation_data,
            num_workers=num_workers,
            num_prefetch=num_prefetch,
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
        )

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        return AutoGluonPredictor(self.autogluonts, prediction_length=self.prediction_length, freq=self.freq)
