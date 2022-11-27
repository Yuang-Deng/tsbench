## Setting up an EC2 instance

Launch an AWS EC2 instance with Ubuntu 20.04 (this avoids potential troubles) with enough disk storage. 
We need at least 500GB to download and upload datasets.

Create an IAM role, called `SagemakerAdmin`, which has the following policies. 
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`

For the `SagemakerAdmin` role, make sure the Trust relationships is like following
to grant SageMaker principal permissions to assume the role:

```angular2html
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        },
        {
            "Sid": "",
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
```
Attach the IAM role `SagemakerAdmin` to the instance. One could do so by doing:

Actions -> Security ->  Modify IAM role -> Select `SagemakerAdmin` -> Update IAM role.


### Config AWS CLI
Before run schedule, we must config the aws cli (only region is needed, they others can be left empty).

```bash
aws configure 
AWS Access Key ID [None]: 
AWS Secret Access Key [None]:
Default region name [None]: us-west-2
Default output format [None]: json
```

## Install
### Clone the package
```bash
git clone https://github.com/Yuang-Deng/tsbench.git
git checkout autogluon_dev
```

### Setting up environment

### Install libraries that are needed for running tsbench   
```bash
bash bin/setup-ec2.sh
source $HOME/.poetry/env
```

### Install python virtual environment through `poetry`
```bash
poetry install
```

After all packages are successfully installed, you can see this line in terminal
```bash
Installing the current project: tsbench (1.0.0)
```

To activate the virtual environment in your terminal:

```bash
poetry shell
```

## Prepare the Data

Before evaluating forecasting methods, you need to prepare the benchmark datasets.
You can run the following commands (assuming that you have executed `poetry shell`):

```bash
# Download and preprocess all datasets
tsbench datasets download

# Upload locally available datasets to your S3 bucket
tsbench datasets upload --bucket <your_bucket_name>
```

Remember the name of the bucket that you used here. You will need it later! 
We don't include Kaggle datasets in this runbook for automation reason. 
If needed, please refer to `README.md` for using Kaggle datasets.

## Prepare AWS Sagemaker

As training jobs on AWS Sagemaker run in Docker containers, you will need to build your own and
upload it to the ECR registry. For this, you must first create an ECR repository named `tsbench`.
Then, you can build and upload it by using the following utility script (it may take up to 1 hour):

```bash
bash bin/build-container.sh
```

The default docker image tag is `autogluon`.

### Build docker image with local autogluon
It may happen that you want to test a version of autogluon that has not been merged.
For this, you need to create a folder named as `thirdparty` under project root 
directory, then go inside `thridparty` folder and put your version of autogluon there.

Then build the docker image with `local` option.
```bash
sh bin/build-container.sh local
```

## Launch Sagemaker job
```bash
tsbench evaluations schedule \
    --config_path configs/benchmark/autogluon_benchmark/autogluon_runbook.yaml \
    --sagemaker_role <your_arn_of_SagemakerAdmin_role> \
    --experiment <your_experiment_name> \
    --data_bucket <your_bucket_name> \
    --data_bucket_prefix <your_dataset_prefix> \
    --output_bucket <your_bucket_name> \
    --output_bucket_prefix <your_output_prefix> \
    --docker_image=tsbench:autogluon \
    --max_runtime=120
```

## Collect the results of sagemaker job and summarize (work in progress)
```bash
tsbench evaluations download \
    --experiment <your_experiment_name> \
    --include_forecasts=False \
    --include_leaderboard=False # only relevant if you want to download leaderboard.csv from autogluon

tsbench evaluations summarize \
    --experiment <your_experiment_name> \
```

# Other things might be helpful
This section includes things that are not generally needed for running the benchmarmks. 
It mostly contains convenient note when development the package.
## For launching without command line options

```python
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        
        {
            "name": "Python: schedule",
            "type": "python",
            "request": "launch",
            "program": "./src/cli/evaluations/schedule.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [
                "--config_path=./configs/benchmark/auto/tsbench_seed.yaml",
                "--sagemaker_role=AmazonSageMaker-ExecutionRole-20210222T141759",
                "--experiment=tsbench-random-seed-exp3",
                "--data_bucket=yuangbucket/tsbench",
                "--data_bucket_prefix=data",
                "--output_bucket=yuangbucket/tsbench",
                "--output_bucket_prefix=evaluations",
                "--docker_image=tsbench-autogluon:jun23_1",
                "--max_runtime=120",
                "--nskip=1"
            ]
        },
        {
            "name": "Python: schedule test",
            "type": "python",
            "request": "launch",
            "program": "./src/cli/evaluations/schedule.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [
                "--config_path=./configs/benchmark/auto/autogluon_test.yaml",
                "--sagemaker_role=AmazonSageMaker-ExecutionRole-20210222T141759",
                "--experiment=tsbench-codelocation-test",
                "--data_bucket=yuangbucket/tsbench",
                "--data_bucket_prefix=data",
                "--output_bucket=yuangbucket/tsbench",
                "--output_bucket_prefix=evaluations",
                "--docker_image=tsbench-autogluon:jun22",
                "--max_runtime=120"
            ]
        },
        {
            "name": "Python: evaluate",
            "type": "python",
            "request": "launch",
            "program": "./src/evaluate.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [
                "--dataset=solar",
                // "--dataset=hospital",
                "--model=autogluon",
                "--autogluon_presets=good_quality",
                "--autogluon_run_time=60"
            ]
        },
        {
            "name": "Python: downlooad metrics",
            "type": "python",
            "request": "launch",
            "program": "./src/cli/evaluations/download.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [
                "--experiment=tsbench-random-seed-exp3",
                // "--experiment=tsbench-leaderboard-test",
                "--include_forecasts=False",
                "--include_leaderboard=False",
                "--format=True",
            ]
        },
        {
            "name": "Python: visualization",
            "type": "python",
            "request": "launch",
            "program": "./src/cli/evaluations/result_visualization.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [
                "--experiment=tsbench-weekend-exp",
            ]
        },
        {
            "name": "Python: dataset download",
            "type": "python",
            "request": "launch",
            "program": "./src/cli/datasets/download_s3.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "args": [
                // "--bucket=yuangbucket/tsbench",
            ]
        },
    ]
}
```


# Integrate Auto-PyTorch
```bash
cd thirdparty
git clone git clone git@github.com:dengdifan/Auto-PyTorch.git
cd Auto-PyTorch
pip install -e .
cd ..
git clone https://github.com/dengdifan/ConfigSpace.git
cd ConfigSpace
pip install .
cd thirdparty/Auto-PyTorch/autoPyTorch
rm -rf automl_common
git clone git@github.com:automl/automl_common.git
cd automl_common
pip install -e .
pip install pytorch_forecasting
```