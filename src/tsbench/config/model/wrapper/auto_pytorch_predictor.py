from typing import Iterator, Optional

import json
from pathlib import Path
import os

import warnings

from gluonts.dataset.common import Dataset
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import QuantileForecast

try:
    from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
except ImportError:
    TimeSeriesForecastingTask = None

try:
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
except ImportError:
    TimeSeriesPredictor = None

AUTOPYTORCH_IS_INSTALLED = TimeSeriesForecastingTask is not None

USAGE_MESSAGE = """
Cannot import `autopytorch`.

The `AutoGluonEstimator` is a thin wrapper for calling the `AutoGluon` package.
"""

class AutoPytorchPredictor(Predictor):

    def __init__(self, model: TimeSeriesForecastingTask, prediction_length: int, freq: str, lead_time: int = 0) -> None:
        super().__init__(prediction_length, freq, lead_time)

        if not AUTOPYTORCH_IS_INSTALLED:
            raise ImportError(USAGE_MESSAGE)

        self.prediction_length = prediction_length
        self.freq = freq
        self.predictor = model
    
    def predict(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
        num_workers: Optional[int] = None,
        num_prefetch: Optional[int] = None,
        **kwargs,
    ) -> Iterator[QuantileForecast]:

        data_frame = TimeSeriesDataFrame(dataset)

        test_sets = self.predictor.dataset.generate_test_seqs()
        try:
            pred = self.predictor.predict(test_sets)
        except Exception as e:
            print(e)
            exit()

        print()

    def deserialize(cls, path: Path, **kwargs) -> "Predictor":
        # predictor = TimeSeriesForecastingTask.load(cls, path)  # type: ignore
        file = path / "metadata.pickle"
        with file.open("r") as f:
            meta = json.load(f)
        return AutoPytorchPredictor(model=None,
            freq=meta["freq"], prediction_length=meta["prediction_length"]
        )

    def serialize(self, path: Path) -> None:
        # self.predictor.save()
        file = path / "metadata.pickle"
        with file.open("w") as f:
            json.dump(
                {
                    "freq": self.freq,
                    "prediction_length": self.prediction_length,
                },
                f,
            )
