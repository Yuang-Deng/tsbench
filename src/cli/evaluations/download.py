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

import os
import tarfile
import tempfile
import pandas as pd
from functools import partial
from pathlib import Path
from typing import Any, cast, Dict, List, Optional
import botocore
import json
import click
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from tsbench.analysis.utils import run_parallel
from tsbench.constants import DEFAULT_EVALUATIONS_PATH
from tsbench.evaluations import aws
from tsbench.evaluations.aws import default_session, TrainingJob
from tsbench.evaluations.tracking.job import Job, load_jobs_from_analysis
from cli.evaluations._main import evaluations
# from ._main import evaluations

BASELINES = ["arima", "ets", "prophet", "mqcnn"]

METRICS = ["mase", "smape", "nrmse", "nd", "ncrps"]

DATASETS = ["m3_yearly", "m3_quarterly", "m3_monthly", "m3_other", "m4_quarterly", "m4_monthly", 
    "m4_weekly", "m4_daily", "m4_hourly", "m4_yearly", "tourism_quarterly", "tourism_monthly", 
    "dominick", "weather", "hospital", "covid_deaths", "electricity", "kdd_2018", "nn5", "rossmann", "solar", "taxi", "wiki"]

@evaluations.command(short_help="Download evaluations to your file system.")
@click.option(
    "--experiment",
    type=str,
    default=None,
    help=(
        "The AWS Sagemaker experiment from which to download the evaluations."
        " If not provided, downloads the publicly available evaluations"
        " (s3://odp-tsbench)."
    ),
)
@click.option(
    "--include_forecasts",
    type=bool,
    default=False,
    help=(
        "Whether to download forecasts (plenty of data) or "
        "only the training, validation and testing metrics."
    ),
)
@click.option(
    "--include_leaderboard",
    type=bool,
    default=False,
    help=(
        "Whether to download leaderboard"
    ),
)
@click.option(
    "--evaluations_path",
    type=click.Path(),
    default=DEFAULT_EVALUATIONS_PATH,
    show_default=True,
    help="The path to which to download the evaluations to.",
)
@click.option(
    "--format",
    type=bool,
    default=False,
    help="Whether to organize the results.",
)
@click.option(
    "--metric",
    type=str,
    default='mase',
    help="Whether to organize the results.",
)
def download(
    experiment: Optional[str], include_forecasts: bool, include_leaderboard: bool, evaluations_path: str, format: bool, metric: str
):
    """
    Downloads either the evaluations of a single AWS Sagemaker experiment or
    the publicly available evaluations.

    The evaluations are downloaded to the provided directory.
    """
    target = Path(evaluations_path)
    target = Path.joinpath(target, experiment)
    target.mkdir(parents=True, exist_ok=True)

    if experiment is None:
        print("Downloading publicly available evaluations...")
        _download_public_evaluations(
            include_forecasts=include_forecasts, evaluations_path=target
        )
        other_jobs = []
    else:
        print(f"Downloading data from experiment '{experiment}'...")
        analysis = aws.Analysis(experiment, status_list=['Completed, Failed'])
        jobs = load_jobs_from_analysis(analysis)
        other_jobs = analysis.other_jobs
        for job in jobs:
            _move_job(job, target=target, include_forecasts=include_forecasts, include_leaderboard=include_leaderboard)
        # process_map(
        #     partial(
        #         _move_job, target=target, include_forecasts=include_forecasts
        #     ),
        #     load_jobs_from_analysis(analysis),
        #     chunksize=1,
        # )
    if format:
        _format(target, metric=metric, experiment=experiment, other_jobs=other_jobs)

def _format(source: Path, experiment: Optional[str], metric:str , other_jobs: List[TrainingJob] = None):
    results = []
    autogluon_models = set()
    models = os.listdir(source)
    #norm results
    for model in models:
        model_dir = Path.joinpath(source, model)
        if Path.is_file(model_dir):
            continue
        datasets = os.listdir(model_dir)
        for ds in datasets:
            # TODO collect dataset we need, try collect all dataset but just print we need
            if ds in DATASETS:
                ds_dir = Path.joinpath(model_dir, ds)
                hyperparameters = os.listdir(ds_dir)
                for hp in hyperparameters:
                    hp_dir = Path.joinpath(ds_dir, hp)
                    config = json.load(open(Path.joinpath(hp_dir, 'config.json'), 'r'))
                    performance = json.load(open(Path.joinpath(hp_dir, 'performance.json'), 'r'))
                    n = len(performance['performances'])
                    res = {}
                    if model == 'autogluon':
                        # leaderboard = pd.read_csv(Path.joinpath(hp_dir, 'leaderboard.csv'))
                        autogluom_model = model + '-' + config['hyperparameters']['presets'] + '-' + str(config['hyperparameters']['run_time'])
                        autogluon_models.add(autogluom_model)
                        res['model'] = autogluom_model
                    else:
                        res['model'] = model
                    res['dataset'] = ds
                    res.update(performance['performances'][-1]['testing'])
                    val_loss = performance['performances'][n-1]['evaluation']['val_loss'] if 'evaluation' in performance['performances'][n-1] else -1
                    res['val_loss'] = val_loss
                    res['seed'] = config['seed']
                    res['hps'] = hp
                    results.append(res)
    
    res_df = pd.DataFrame(results)
    res_df = res_df.loc[res_df.groupby(['dataset', 'model', 'seed']).val_loss.idxmin()]
    index_models = BASELINES + list(autogluon_models)
    print(res_df.pivot_table(index='dataset', columns='model', values=metric).reindex(index_models, axis=1))

    # abnormal results
    abnormal_results = []
    if len(other_jobs) > 0:
        for job in other_jobs:
            res = {}
            res.update(job.hyperparameters)
            res['status'] = job.status
            abnormal_results.append(res)
            print(res['model'], ' \t', res['dataset'], ' \t', res['status'])

    # TODO leader_board
    
    if experiment is None:
        json.dump(results, open(Path.joinpath(source, 'tsbench.json'), 'w+'), indent=2)
    else:
        json.dump(results, open(Path.joinpath(source, experiment + '.json'), 'w+'), indent=2)
        print('results of complemented experiments is saved in', Path.joinpath(source, experiment + '.json'))
        json.dump(abnormal_results, open(Path.joinpath(source, experiment + '-abnormal.json'), 'w+'), indent=2)
        print('results of others experiments is saved in', Path.joinpath(source, experiment + '-abnormal.json'))

def _download_public_evaluations(
    include_forecasts: bool, evaluations_path: Path
) -> None:
    public_bucket = "odp-tsbench"
    session = default_session()
    client = session.client(
        "s3",
        config=botocore.client.Config(  # type: ignore
            signature_version=botocore.UNSIGNED,
            max_pool_connections=2 * cast(int, os.cpu_count()),
        ),
    )

    # First, download the metrics
    print("Downloading metrics...")
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / "metrics.tar.gz"
        client.download_file(public_bucket, "metrics.tar.gz", str(file))
        with tarfile.open(file, mode="r:gz") as tar:
            tar.extractall(evaluations_path)

    # Then, optionally download the forecasts
    if include_forecasts:
        print("Downloading forecasts...")

        # First, get all files
        with tqdm(desc="List objects") as progress:
            response = client.list_objects(Bucket=public_bucket)
            objects = _extract_object_names(response)
            progress.update()
            while response["IsTruncated"]:
                response = client.list_objects(
                    Bucket=public_bucket, Marker=objects[-1]
                )
                objects.extend(_extract_object_names(response))
                progress.update()

        # Then, download all of the objects
        run_parallel(
            partial(
                _download_object,
                bucket=public_bucket,
                client=client,
                destination=evaluations_path,
            ),
            objects,
            2 * cast(int, os.cpu_count()),
        )


def _download_object(
    key: str, bucket: str, client: Any, destination: Path
) -> None:
    target = destination / key
    target.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(Bucket=bucket, Key=key, Filename=str(target))


def _extract_object_names(response: Dict[str, Any]) -> List[str]:
    return [
        obj["Key"]
        for obj in response["Contents"]
        if not obj["Key"].endswith("/")
        and obj["Key"] != "metrics.tar.gz"
        and not obj["Key"].endswith("config.json")
        and not obj["Key"].endswith("performance.json")
    ]


def _move_job(job: Job, target: Path, include_forecasts: bool, include_leaderboard: bool):
    job.save(target, include_forecasts=include_forecasts, include_leaderboard=include_leaderboard)

# download()