# AWS enviroment
AWS CLI configuration, refer to https://quip-amazon.com/uG7bAb0veru6/tsbench-setup

# install
### autogluon
```bash 
got clone https://github.com/awslabs/autogluon.git 
sh full_install.sh
```
### gluonts 
```bash
got clone https://github.com/awslabs/gluon-ts.git
pip install -e .
```
### tsbench
```bash
git cloen https://github.com/Yuang-Deng/tsbench.git
poetry install
```

# build docker image and upload to ecr
the dockrfile is modified by me to install autogluon in docker,
before build docker, you may need to create a repository in ECR, and set a tag for your image
```bash
sh bin/build-container.sh
```

# I use vscode to run evaluate and schedule script, this is my launch.json
the sagemaker role may need to modified
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
                "--experiment=tsbench-autogluon",
                "--data_bucket=yuangbucket/tsbench",
                "--data_bucket_prefix=data",
                "--output_bucket=yuangbucket/tsbench",
                "--output_bucket_prefix=evaluations",
                "--docker_image=tsbench-autogluon:jun14",
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
                // "--dataset=m4_hourly",
                // "--dataset=m4_yearly",
                // "--dataset=wind_farms",
                "--dataset=london_smart_meters",
                // "--dataset=vehicle_trips",
                "--model=autogluon"
            ]
        }
    ]
}
```