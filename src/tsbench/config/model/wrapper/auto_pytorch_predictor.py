from typing import Iterator, Optional

import json
from pathlib import Path

from gluonts.dataset.common import Dataset
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import SampleForecast
import warnings

# try:
#     from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
#     from autoPyTorch.datasets.time_series_dataset import TimeSeriesSequence
# except ImportError:
#     TimeSeriesForecastingTask = None

from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
from autoPyTorch.datasets.time_series_dataset import TimeSeriesSequence

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

        # TODO a quickly method?
        sequences_kwargs = {"freq": self.freq,
            "time_feature_transform": self.predictor.dataset.time_feature_transform,
            "train_transforms": self.predictor.dataset.train_transform,
            "val_transforms": self.predictor.dataset.val_transform,
            "n_prediction_steps": self.prediction_length,
            "sp": self.predictor.dataset.seasonality,
            "known_future_features": 0,
            "static_features": None}
        
        self.pred = []
        all_test_ds = self.predictor.dataset.generate_test_seqs()
        for ds, test_ds in zip(dataset, all_test_ds):
            ts = TimeSeriesSequence(
                X=None,
                Y=ds["target"][:, None],
                start_time_train=ds["start"],
                X_test=None,
                Y_test=None,
                start_time_test=None,
                only_has_past_targets=True,
                time_features=self.predictor.dataset.compute_time_features([ds["start"]], [len(ds["target"])]),
                **sequences_kwargs)
            pred = self.predictor.predict([ts])
            pred_test = self.predictor.predict([test_ds])
            self.pred.append(pred)
        # for samples in pred:
            yield SampleForecast(
                samples=pred,
                start_date=ds["start"],
                freq=self.freq,
                item_id=ds["item_id"],
            )

    def leaderboard(
        self,
        dataset: Dataset,
        **kwargs,
    ) -> None:
        all_test_ds = self.predictor.dataset.generate_test_seqs()
        test_series_list = []
        for ds in dataset:
            test_series_list.append(ds["target"][-self.prediction_length:])
        res = self.compute_loss(self.prediction_length, seasonality=1, final_forecasts=self.pred, test_series_list=dataset, train_series_list=all_test_ds)
        print(res)
        

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

    def compute_loss(self, forecast_horizon, seasonality, final_forecasts, test_series_list, train_series_list) -> dict:
        import numpy as np
        epsilon = 0.1

        MASE = []
        sMAPE = []
        msMAPE = []
        MAE = []
        RMSE = []

        sqrt_forecast_horizon = np.sqrt(forecast_horizon)

        idx = 0

        for f, y, y_data in zip(final_forecasts, test_series_list, train_series_list):

            y = y["target"][-forecast_horizon:]
            y_data = y_data.Y
            M = len(y_data)

            diff_abs = np.abs(f - y)

            if M == seasonality:
                mase_denominator = 0
            else:
                mase_denominator_coefficient = forecast_horizon / (M - seasonality)
                mase_denominator = mase_denominator_coefficient * \
                                np.sum(np.abs(y_data[seasonality:] - y_data[:-seasonality]))

                abs_loss = np.sum(diff_abs)
                mase = abs_loss / mase_denominator

            if mase_denominator == 0:
                mase_denominator_coefficient = forecast_horizon / (M - 1)
                mase_denominator = mase_denominator_coefficient * \
                                np.sum(np.abs(y_data[1:] - y_data[:-1]))
                mase = abs_loss / mase_denominator

            if np.isnan(mase) or np.isinf(mase):
                # see the R file
                pass
            else:
                MASE.append(mase)

            smape = 2 * diff_abs / (np.abs(y) + np.abs(f))
            smape[diff_abs == 0] = 0
            smape = np.sum(smape) / forecast_horizon
            sMAPE.append(smape)

            msmape = np.sum(2 * diff_abs / (np.maximum(np.abs(y) + np.abs(f) + epsilon, epsilon + 0.5))) / forecast_horizon
            msMAPE.append(msmape)

            mae = abs_loss / forecast_horizon
            MAE.append(mae)

            rmse = np.linalg.norm(f - y) / sqrt_forecast_horizon
            RMSE.append(rmse)


            idx += 1
        res = {}

        res['Mean MASE'] = np.mean(MASE)

        res['Median MASE'] = np.median(MASE)

        res['Mean sMAPE'] = np.mean(sMAPE)
        res['Median sMAPE'] = np.median(sMAPE)

        res['Mean mSMAPE'] = np.mean(msMAPE)
        res['Median mSMAPE'] = np.median(msMAPE)

        res['Mean MAE'] = np.mean(MAE)
        res['Median MAE'] = np.median(MAE)

        res['Mean RMSE'] = np.mean(RMSE)
        res['Median RMSE'] = np.median(RMSE)


        return res
