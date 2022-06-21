# clone repository and conda 
```bash
git clone https://github.com/Yuang-Deng/tsbench.git
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
```

# install on EC2
```bash
sh Anaconda3-2022.05-Linux-x86_64.sh 
conda create -n tsbench python=3.8
conda activate tsbench
sudo apt-get update # this is necessary after jun17, maybe the server has change ip? If no this command, an error "E: Failed to fetch http://us-west-2.ec2.archive.ubuntu.com/ubuntu/pool/main/l/linux/linux-libc-dev_5.4.0-109.123_amd64.deb  404  Not Found [IP: 34.212.136.213 80]" will be raised

# sudo apt-get install gcc # this will be unnecessary after sudo apt-get update

cd tsbench 
git checkout autogluon_dev  # the latest version of code is in branch autogluon_dev
pip install poetry
poetry install

# this will be not necessary, please install the packages which can not successfully installed by poetry
pip install sagemaker
...
```



## run schedule to launch job in sagemaker
### AWS CLI config
before run schedule, we must config the aws cli.

In aws configure, only region is needed, they others can be left empty.
```bash
sudo apt install awscli
aws configure
AWS Access Key ID [None]: your id
AWS Secret Access Key [None]: your key
Default region name [None]: us-west-2
Default output format [None]: json
```
### docker image build
the dockrfile is modified by me to install autogluon in docker,
before build docker, you may need to create a repository in ECR, and set a tag for your image

See line 12 in bin/build-container.sh. Before the colon, it was your ECR repository name. After the colon, is the tag of your image.

for example
```bash
docker build \
    -t $REGISTRY/tsbench-autogluon:jun20localtest \
    -f Dockerfile .
```
*tsbench-autogluon* is the name of you ECR repository and *jun20localtest* is the tag of the image

```bash
sh bin/build-container.sh
```

The install of autogluon on docker is install manually, because use poetry to install the latest version of autogluon with source code will need setup.py, but the autogluon has no setup.py, if want use poetry install it, we may need wait the latest version of autogluon released as wheel package.

When build docker image, it will clone the latest version of autogluon

if you want modify autogluon locally and build docker image with your code, you need to create a thirdparty folder to store the repository of thirdparty, now just autogluon store in there, the docker image will copy the code in thirdparty and install it.

```bash
cd tsbench
mkdir thirdparty
git clone https://github.com/awslabs/autogluon.git thirdparty/autogluon
cd thirdparty/autogluon
./full_install.sh
sh bin/build-container.sh local   # add local parameter will use Dockerfile_local to build image
```

### an issue
when run python scripts blow, you need to add one line on the script which you want to run to invoke the function, when you run other scripts, you must delet the line you added before, otherwise there will be some errors.

### run
```bash
python ./src/cli/evaluations/schedule.py \
    --config_path=./configs/benchmark/auto/autogluon.yaml \
    --sagemaker_role=AmazonSageMaker-ExecutionRole-20210222T141759 \
    --experiment=tsbench-autogluon-runbook-test \
    --data_bucket=yuangbucket/tsbench \
    --data_bucket_prefix=data \
    --output_bucket=yuangbucket/tsbench \
    --output_bucket_prefix=evaluations \
    --docker_image=tsbench-autogluon:jun17 \
    --max_runtime=120 \
```

## run evaluate to debug locally with single dataset
### dataset download
before run evaluate locally, the dataset must downloaded and save at /home/ubuntu/data/datasets
the datasets is upload to my s3 bucket, s3://yuangbucket/tsbench/data/
### use latest version of autogluon
I will try to install latest version from source code use poetry, but now, we need install the latest version manually.
```bash
git clone https://github.com/awslabs/autogluon.git
./full_install.sh
```
### run
```bash
python ./src/evaluate.py \
    --dataset=m4_hourly \
    --model=autogluon \
```

## run locally
create a folder to clone autogluon code and install it.
```bash
cd tsbench
mkdir thirdparty
git clone https://github.com/awslabs/autogluon.git thirdparty/autogluon
cd thirdparty/autogluon
./full_install.sh
```
run
```bash
python ./src/evaluate.py \
    --dataset=m1_yearly \
    --model=autogluon \
```

## if use vscode, this is my launch.json
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
                "--config_path=./configs/benchmark/auto/autogluon.yaml",
                "--sagemaker_role=AmazonSageMaker-ExecutionRole-20210222T141759",
                "--experiment=tsbench-rerun-dataset",
                "--data_bucket=yuangbucket/tsbench",
                "--data_bucket_prefix=data",
                "--output_bucket=yuangbucket/tsbench",
                "--output_bucket_prefix=evaluations",
                "--docker_image=tsbench-autogluon:jun17",
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
                "--dataset=m1_yearly",
                "--model=autogluon"
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
                "--experiment=tsbench-weekend-exp",
                "--include_forecasts=False",
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
        }
    ]
}
```

## some issue with error raised with failed poetry install
after run poetry install, use pip list to check if the tsbench is installed 
if tsbench is not installed, but run evaluate and schedule need tsbench, we can delete the line after line 10, and use poetry install


# collect tsbench result
the tsbench result collect script is collect result by the experiment name of sagemaker job, the experiment name is a parameter of schedule, an experiment will run multiple different configurat, it correspond to multiple job on sagemaker, this job will have the same experiment name but with different suffix, this script will collect result by experiment name, ignore the suffix.

```bash
python ./src/cli/evaluations/download.py \
    --experiment=tsbench-weekend-exp \
    --include_forecasts=False \
    --format=True \
```

if format=True, the results will be print as a table in terminal, the abnormal results will be print as json in terminal, all results will be store as json file.

if the results are stored as json file, the result_visualization can be used to visualiza the results again.

```bash
python ./src/cli/evaluations/result_visualization.py \
    --experiment=tsbench-weekend-exp 
```

# modify autogluon locally and build docker image
create a thirdparty folder to store the repository of thirdparty, now just autogluon, the docker image can be build successfully
```bash
cd tsbench
mkdir thirdparty
git clone https://github.com/awslabs/autogluon.git thirdparty/autogluon
cd thirdparty/autogluon
./full_install.sh
```