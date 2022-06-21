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

import json
from pathlib import Path
from typing_extensions import Required
from unittest import result
import click
import os
import pandas as pd
from tsbench.constants import DEFAULT_EVALUATIONS_PATH
from tsbench.utils import compress_directory
from cli.evaluations._main import evaluations
from cli.evaluations.download import BASELINES, METRICS, DATASETS
# from ._main import evaluations


@evaluations.command(
    short_help="Archive metrics of all evaluations into a single file."
)
@click.option(
    "--evaluations_path",
    type=click.Path(exists=True),
    default=DEFAULT_EVALUATIONS_PATH,
    help="The directory where TSBench evaluations are stored.",
)
@click.option(
    "--experiment",
    type=click.Path(),
    required=True,
    help="The experiment name which you want to visualization.",
)
def result_visualization(evaluations_path: str, experiment: str):
    """
    Archives the metrics of all evaluations found in the provided directory
    into a single file.

    This is most probably only necessary for publishing the metrics in an
    easier format.
    """
    source = Path(evaluations_path)
    results_path = Path.joinpath(source, experiment + '.json')
    abnormal_results_path = Path.joinpath(source, experiment + '-abnormal.json')

    exp_model = 'autogluon'
    results = json.load(open(results_path, 'r'))
    res_df = pd.DataFrame(results)
    metric = 'smape'
    exp_models = set()
    for res in results:
        if exp_model in res['model']:
            exp_models.add(res['model'])
    index_models = BASELINES + list(exp_models)
    print(res_df.pivot_table(index='dataset', columns='model', values=metric).reindex(index_models, axis=1))

    abnormal_results = json.load(open(abnormal_results_path, 'r'))
    print()
    print('model \t\t', 'dataset \t\t', 'status')
    for res in abnormal_results:
        print(res['model'], ' \t\t', res['dataset'], ' \t\t', res['status'])

result_visualization()