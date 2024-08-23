FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 AS base

ARG PYTHON_VERSION=3.10

# set arguments
# ARG SENTRY_DSN

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# set environment variables
# ENV SENTRY_DSN=${SENTRY_DSN}

# for python
RUN apt-get update -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install -y python${PYTHON_VERSION}

# update and install packages
RUN apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    awscli \
    zip \
    unzip \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python${PYTHON_VERSION}-distutils \
    software-properties-common \
    vim \
    && \
    rm -rf /var/lib/apt/lists/*

# 크롤링시 아래 코드 추가
# RUN wget -nc https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && apt update && apt install -y ./google-chrome-stable_current_amd64.deb

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2
RUN python3 -m pip install --upgrade pip==20.0.2

# set working directory
WORKDIR /workspace

# install python packages
ADD requirements.txt /workspace/
RUN pip install --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple -r requirements.txt
RUN apt-get update
RUN apt-get install python3.10-dev -y
# install apt packages

COPY . /workspace

RUN echo $(python3 --version)

RUN apt-get update
# 실행 command에 맞게 유동적으로 변경
# $ ./command_demo.sh가 로컬 커맨드라면 CMD ["/bin/bash", "/workspace/command_demo.sh"] (run.py에 작업 권장)
CMD ["python3", "/workspace/run.py"]