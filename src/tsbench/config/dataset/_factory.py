# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from collections.abc import Iterable
import itertools

import pandas as pd
from pathlib import Path
from typing import Dict, Type, TypeVar, Union
from tsbench.constants import DEFAULT_DATA_PATH
from ._base import DatasetConfig

DATASET_REGISTRY: Dict[str, Type[DatasetConfig]] = {}

ITEMID = "item_id"
TIMESTAMP = "timestamp"


D = TypeVar("D", bound=Type[DatasetConfig])


def register_dataset(cls: D) -> D:
    """
    Registers the provided class in the global dataset registry.
    """
    DATASET_REGISTRY[cls.name()] = cls
    return cls


def get_dataset_config(
    name: str, path: Union[Path, str] = DEFAULT_DATA_PATH
) -> DatasetConfig:
    """
    This method creates the dataset configuration of the model with the
    specified name.

    Args:
        name: The canonical name of the dataset. See `DATASET_REGISTRY`.
        path: The root of the dataset directory.

    Returns:
        The dataset configuration.
    """
    # Get the dataset
    assert name in DATASET_REGISTRY, f"Dataset name '{name}' is unknown."
    return DATASET_REGISTRY[name](Path(path))

def construct_pandas_frame_from_iterable_dataset(
    iterable_dataset: Iterable
) -> pd.DataFrame:
    _validate_iterable(iterable_dataset)

    all_ts = []
    id_set = set()
    for i, ts in enumerate(iterable_dataset):
        # print(ts['item_id'])
        id_set.add(ts['item_id'])
        start_timestamp = ts["start"]
        target = ts["target"]
        datetime_index = tuple(
            pd.date_range(
                start_timestamp, periods=len(target), freq=start_timestamp.freq
            )
        )
        idx = pd.MultiIndex.from_product(
            [(i,), datetime_index], names=[ITEMID, TIMESTAMP]
        )
        ts_df = pd.Series(target, name="target", index=idx).to_frame()
        all_ts.append(ts_df)
    print(len(id_set))
    return pd.concat(all_ts)

def _validate_iterable(data: Iterable):
    if not isinstance(data, Iterable):
        raise ValueError("data must be of type Iterable.")

    first = next(iter(data), None)
    if first is None:
        raise ValueError("data has no time-series.")

    for i, ts in enumerate(itertools.chain([first], data)):
        if not isinstance(ts, dict):
            raise ValueError(
                f"{i}'th time-series in data must be a dict, got{type(ts)}"
            )
        if not ("target" in ts and "start" in ts):
            raise ValueError(
                f"{i}'th time-series in data must have 'target' and 'start', got{ts.keys()}"
            )
        if not isinstance(ts["start"], pd.Timestamp) or ts["start"].freq is None:
            raise ValueError(
                f"{i}'th time-series must have timestamp as 'start' with freq specified, got {ts['start']}"
            )
