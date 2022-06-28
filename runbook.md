# clone repository and conda 
```bash
git clone https://github.com/Yuang-Deng/tsbench.git
git checkout autogluon_dev
```

# install on EC2
## enviroment set
```bash
bash bin/setup-ec2.sh  #must bash, if use sh, an error will be encountered
source $HOME/.poetry/env

poetry install
poetry shell
```
When you need use tsbench cmd to run program, you need use source $HOME/.poetry/env to activate the virtual enviroment, and poetry shell to make cmd of tsbench available
### issue
You may encounter some errors during poetry install, you can see the log in terminal and install the package manually which not installed successful.

After all packages are successfully installed, you can see this line in terminal
```bash
Installing the current project: tsbench (1.0.0)
```

## config AWS CLI
Before run schedule, we must config the aws cli.

<!-- In aws configure, only region is needed, they others can be left empty. -->
```bash
aws configure 
AWS Access Key ID [None]: your id
AWS Secret Access Key [None]: your key
Default region name [None]: us-west-2
Default output format [None]: json
```

## config kaggle
use your kaggle account

## download datasets

```bash
tsbench datasets download \
    --path=/home/ubuntu/data/datasets \ # the path you want to store the datasets.
    --dataset=tourism_quarterly \ # specific a dataset which you want to download, if not specific, all datasets will be downloaded.
```

if you do not config the kaggle account, some dataset will not be download, but we can use the dataset 

if "Command 'tsbench' not found", use poetry shell to activate the virtual enviroment

## upload dataset
upload the datasets to s3 bucket
```bash
tsbench datasets upload \
    --bucket=yuangbucket \ # your s3 bucket
    --path=/home/ubuntu/data/datasets \ # The path of the datasets you downloaded
    --prefix=datatest \ # the path you want to upload to in s3 bucket
```

## build docker and upload to ECR
Since autogluon is still under development, here are two ways to build docker images.

Before run script to build docker image, you need to create a ECR repository, The name of the repository must be the same as the tag of docker you build

for instance
```bash
docker build \
    -t $REGISTRY/tsbench-autogluon:jun22 \
    -f $DOCKERFILE_PATH . 
```
tsbench-autogluon is the name of the ECR repository and jun22 is the tag of this image. This can be set in bin/build-container.sh

### build docker image with local code
You need to create a folder named as thirdparty, download the repository which you need in this folder, and modify the Dockerfile_local to install it on docker image
```bash
sh bin/build-container.sh local
```

### build docker image with remote code
You can modify the Dockerfile to clone git repository and install it on docker image
```bash
sh bin/build-container.sh
```

## launch sagemaker job
```bash
tsbench evaluations schedule \
    --config_path=./configs/benchmark/auto/autogluon.yaml \ # the config file you want to run
    --sagemaker_role=AmazonSageMaker-ExecutionRole-20210222T141759 \ # your sagemaker role
    --experiment=tsbench-autogluon-runbook-test \ # the name of experiment
    --data_bucket=yuangbucket \ # your s3 bucket name
    --data_bucket_prefix=datatest \ # the path of datasets in your s3 bucket
    --output_bucket=yuangbucket \ # your s3 bucket name
    --output_bucket_prefix=evaluations \ # the path of the results you want to store in your s3 bucket
    --docker_image=tsbench-autogluon:jun17 \ # the docker repository and tag you build before
    --max_runtime=120 \
```

## collect the results of sagemaker job
```bash
tsbench evaluations download \
    --experiment=tsbench-autogluon-runbook-test \ # the experiment name you want to download
    --include_forecasts=False \ 
    --include_leaderboard=False \ # download leaderboard to dictory, it only valid when model is autogluon
    --format=True \ # whether to visualize the results as a table
```


# for developer
this is my launch.json

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