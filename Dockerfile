FROM ghcr.io/josephgiovanelli/mo-importance:0.0.6

RUN apt-get update && \
    apt-get install -y git --no-install-recommends

RUN pip install --upgrade pip && \
    pip install --upgrade pip setuptools wheel

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt && \
    rm requirements.txt

RUN mkdir interactive-mo-ml
WORKDIR /home/interactive-mo-ml
ENTRYPOINT ["./scripts/wrapper_experiments.sh"]
