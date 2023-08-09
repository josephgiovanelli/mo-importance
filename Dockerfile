FROM ghcr.io/josephgiovanelli/mo-importance:0.0.6

RUN mkdir dump
WORKDIR /home/dump
COPY . .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
RUN chmod 777 scripts/*

WORKDIR /home
RUN mkdir interactive-mo-ml
WORKDIR /home/interactive-mo-ml
ENTRYPOINT ["./scripts/wrapper_experiments.sh"]
