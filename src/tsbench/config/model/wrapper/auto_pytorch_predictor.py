from typing import Iterator, Optional

import json
from pathlib import Path

from gluonts.dataset.common import Dataset
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import SampleForecast

try:
    from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
    from autoPyTorch.datasets.time_series_dataset import TimeSeriesSequence
except ImportError:
    TimeSeriesForecastingTask = None

AUTOPYTORCH_IS_INSTALLED = TimeSeriesForecastingTask is not None

USAGE_MESSAGE = """
Cannot import `autopytorch`.

The `AutoPytorchPredictor` is a thin wrapper for calling the `AutoGluon` package.
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
    ) -> Iterator[SampleForecast]:
        predict_ts = []
        for ds in dataset:
            ts = TimeSeriesSequence(
                X=None,
                Y=ds["target"][:, None],
                start_time=ds["start"],
                freq=self.freq,
                time_feature_transform=self.predictor.dataset.time_feature_transform,
                train_transforms=self.predictor.dataset.train_transform,
                val_transforms=self.predictor.dataset.val_transform,
                n_prediction_steps=self.prediction_length,
                sp=self.predictor.dataset.seasonality,
                is_test_set=True)
            predict_ts.append(ts)
        pred = self.predictor.predict(predict_ts)
        for p, ds in zip(pred, dataset):
            yield SampleForecast(
                samples=p[:, None],
                start_date=ds["start"],
                freq=self.freq,
                item_id=ds["item_id"],
            )
        

    def deserialize(cls, path: Path, **kwargs) -> "Predictor":
        file = path / "metadata.pickle"
        with file.open("r") as f:
            meta = json.load(f)
        return AutoPytorchPredictor(model=None,
            freq=meta["freq"], prediction_length=meta["prediction_length"]
        )

    def serialize(self, path: Path) -> None:
        file = path / "metadata.pickle"
        with file.open("w") as f:
            json.dump(
                {
                    "freq": self.freq,
                    "prediction_length": self.prediction_length,
                },
                f,
            )