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

from typing import Dict, List
from .performance import Performance


def metric_definitions() -> List[Dict[str, str]]:
    """
    Returns the metric definitions to be used to collect performance measures
    from AWS Sagemaker.
    """
    # pylint: disable=no-member
    scalar_metrics = [
        {"Name": name, "Regex": _metric_regex(name)}
        for name in Performance.__dataclass_fields__  # type: ignore
    ]
    list_metrics = [
        {
            "Name": "train_loss",
            "Regex": f"'epoch_loss'={_FLOATING_POINT_REGEX}",
        },
        {
            "Name": "val_loss",
            "Regex": f"'validation_epoch_loss'={_FLOATING_POINT_REGEX}",
        },
        {
            "Name": "val_ncrps",
            "Regex": _metric_regex("val_ncrps"),
        },
        {
            "Name": "val_nd",
            "Regex": _metric_regex("val_nd"),
        },
        {
            "Name": "val_nrmse",
            "Regex": _metric_regex("val_nrmse"),
        },
        {
            "Name": "val_mase",
            "Regex": _metric_regex("val_mase"),
        },
        {
            "Name": "val_smape",
            "Regex": _metric_regex("val_smape"),
        },
        {
            "Name": "val_latency",
            "Regex": _metric_regex("val_latency"),
        },
    ]
    custimze_metrics = [
        {
            "Name": "autogluon_traing_time",
            "Regex": f"Total runtime: {_FLOATING_POINT_REGEX}"
        },
        {
            "Name": "autopytorch_traing_time",
            "Regex": f"autopytorch runtime: {_FLOATING_POINT_REGEX}"
        },

    ]
    return scalar_metrics + list_metrics + custimze_metrics


# -------------------------------------------------------------------------------------------------

_FLOATING_POINT_REGEX = (
    r"(([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]\d+)?)|[Nn][Aa][Nn])"
)


def _metric_regex(target: str) -> str:
    return f"tsbench\\[{target}\\]: {_FLOATING_POINT_REGEX}"
