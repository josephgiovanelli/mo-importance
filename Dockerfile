FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.9

RUN apt-get update && \
    apt-get install -y git --no-install-recommends

RUN pip install --upgrade pip && \
    pip install --upgrade pip setuptools wheel

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r /requirements.txt && \
    rm requirements.txt

RUN wget -c https://github.com/slds-lmu/yahpo_data/archive/refs/tags/v1.0.zip && \
    unzip v1.0.zip && \
    rm -rf v1.0.zip

WORKDIR /home
