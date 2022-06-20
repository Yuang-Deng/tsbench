FROM python:3.8.12-buster

# Install R
RUN apt-get update \
    && apt-get install -y r-base \
    && R -e 'install.packages(c("forecast", "nnfor"), repos="https://cloud.r-project.org")'

# Install project dependencies
RUN pip install poetry==1.1.6 \
    && poetry config virtualenvs.create false
COPY poetry.lock pyproject.toml /dependencies/
RUN cd /dependencies \
    && poetry install --no-dev --no-root --no-interaction --no-ansi

# the latest version of autogluon depend on mxnet==1.9, I will try to install mxnet by poetry
RUN pip uninstall mxnet -y
RUN pip install mxnet==1.9

# install autogluon with locally code
RUN pip uninstall autogluon -y
COPY thirdparty/autogluon /dependencies/autogluon/
WORKDIR /dependencies/autogluon/
RUN ./full_install.sh

# RUN pip uninstall autogluon -y
# # TODO fix the autogluon version
# RUN git clone https://github.com/awslabs/autogluon.git
# WORKDIR /autogluon/
# RUN ./full_install.sh