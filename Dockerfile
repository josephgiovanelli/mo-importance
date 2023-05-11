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
    rm requirements.txt
RUN pip install git+https://github.com/kiudee/cs-ranking.git
RUN pip install h5py
RUN pip install "git+https://github.com/slds-lmu/yahpo_gym#egg=yahpo_gym&subdirectory=yahpo_gym"
    # git clone https://github.com/automl/HPOBench.git && \
    # cd HPOBench && \
    # pip install . 

RUN wget -c https://github.com/slds-lmu/yahpo_data/archive/refs/tags/v1.0.zip && \
    unzip v1.0.zip && \
    rm -rf v1.0.zip
    
WORKDIR /home

