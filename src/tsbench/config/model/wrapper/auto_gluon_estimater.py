import logging
from time import time
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional

import numpy as np
import pandas as pd
import toolz

from mxnet.gluon import HybridBlock

from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.mx.trainer.callback import Callback, CallbackList

from gluonts.dataset.common import Dataset
from gluonts.dataset.loader import DataLoader
from gluonts.itertools import Cached
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor
from gluonts.mx.trainer import Trainer
from gluonts.transform import Transformation

from .auto_gluon_predictor import AutoGluonPredictor

try:
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
except ImportError:
    TimeSeriesPredictor = None

AUTOGLUON_IS_INSTALLED = TimeSeriesPredictor is not None

USAGE_MESSAGE = """
Cannot import `autogluon`.

The `ProphetPredictor` is a thin wrapper for calling the `fbprophet` package.
In order to use it you need to install it using one of the following two
methods:

    # 1) install fbprophet directly
    pip install fbprophet

    # 2) install gluonts with the Prophet extras
    pip install gluonts[Prophet]
"""

class AutoGluonEstimator(Estimator):
    """
    Wrapper around `Autogluon <https://github.com/facebook/prophet>`_.

    The `AutoGluonPredictor` is a thin wrapper for calling the `Autogluon`
    package. In order to use it you need to install the package::

        # you can either install Prophet directly
        pip install fbprophet

        # or install gluonts with the Prophet extras
        pip install gluonts[Prophet]

    Parameters
    ----------
    freq
        Time frequency of the data, e.g. '1H'
    prediction_length
        Number of time points to predict
    prophet_params
        Parameters to pass when instantiating the prophet model.
    init_model
        An optional function that will be called with the configured model.
        This can be used to configure more complex setups, e.g.

        >>> def configure_model(model):
        ...     model.add_seasonality(
        ...         name='weekly', period=7, fourier_order=3, prior_scale=0.1
        ...     )
        ...     return model
    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        autogluonts_params: Optional[Dict] = None,
        presets: Optional[str] = None,
        time_limit: Optional[float] = None,
        hyperparameters: Optional[str] = None,
        # callbacks: Optional[List[Callback]] = None,
    ) -> None:
        super().__init__(freq=freq, prediction_length=prediction_length)

        if not AUTOGLUON_IS_INSTALLED:
            raise ImportError(USAGE_MESSAGE)

        if autogluonts_params is None:
            autogluonts_params = {}

        assert "uncertainty_samples" not in autogluonts_params, (
            "Parameter 'uncertainty_samples' should not be set directly. "
            "Please use 'num_samples' in the 'predict' method instead."
        )
        self.freq = freq
        self.prediction_length = prediction_length
        self.autogluonts_params = autogluonts_params
        self.autogluonts = TimeSeriesPredictor(prediction_length=prediction_length)
        self.presets = presets
        self.time_limit = time_limit
        self.hyperparameters = hyperparameters

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
        # train_dataframe.index.levels[1].freq = self.freq
        # print(train_dataframe.freq)
        # train_dataframe.freq = self.freq

        print(f"*** calling autogluon with presets={self.presets}, time_limit={self.time_limit}")
        tspredictor = self.autogluonts.fit(train_dataframe, tuning_data=valid_dataframe, presets=self.presets, time_limit=self.time_limit)
        # tspredictor = self.autogluonts.fit(train_dataframe, tuning_data=valid_dataframe, hyperparameters=self.hyperparameters, time_limit=60)

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
