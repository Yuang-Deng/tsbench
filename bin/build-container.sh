#!/bin/bash
set -e

echo "Fetching credentials from AWS..."
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region)
ECR_PASSWORD=$(aws ecr get-login-password)
REGISTRY=$AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com

DOCKERFILE_PATH="Dockerfile"
if [ "$1" = "local" ];
then
    echo "Build docker image with local autogluon"
    DOCKERFILE_PATH="Dockerfile_local"
else
    echo "Build docker image with remote autogluon"     
fi

echo "Building image..."
docker build \
    -t $REGISTRY/tsbench:autogluon \
    -f $DOCKERFILE_PATH . 

echo "Logging in to ECR..."
echo $ECR_PASSWORD | \
    docker login --username AWS --password-stdin $REGISTRY

echo "Pushing image..."
docker push $REGISTRY/tsbench:autogluon
