FROM ghcr.io/josephgiovanelli/mo-importance:0.0.6

RUN mkdir interactive-mo-ml
WORKDIR /home/interactive-mo-ml
COPY . .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
RUN chmod 777 scripts/*

ENTRYPOINT ["./scripts/wrapper_experiments.sh"]
