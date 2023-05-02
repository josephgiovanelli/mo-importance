#!/bin/bash
docker stop mo-importance
docker rm mo-importance
docker run --name mo-importance --volume $(pwd):/home --detach -t ghcr.io/josephgiovanelli/mo-importance:$1
docker exec mo-importance bash ./scripts/wrapper_experiments.sh