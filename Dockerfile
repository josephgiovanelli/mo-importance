FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.9
RUN apt-get update && \
    apt-get install -y git --no-install-recommends
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install black && \
    pip install --no-cache-dir --upgrade -r /requirements.txt && \
    pip install requests && \
    pip install tabulate && \
    pip install future && \
    rm requirements.txt && \
    pip install git+https://github.com/kiudee/cs-ranking.git && \
    pip install h5py
WORKDIR /home
