FROM ghcr.io/josephgiovanelli/mo-importance:0.0.6

RUN apt-get update && \
    apt-get install -y git --no-install-recommends

RUN pip install --upgrade pip && \
    pip install --upgrade pip setuptools wheel

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt && \
    rm requirements.txt

RUN wget -c https://github.com/slds-lmu/yahpo_data/archive/refs/tags/v1.0.zip && \
    unzip v1.0.zip && \
    rm -rf v1.0.zip

RUN cd home && mkdir interactive-mo-ml
WORKDIR /home/interactive-mo-ml
ENTRYPOINT ["./scripts/wrapper_experiments.sh"]
