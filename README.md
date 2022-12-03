# TTS-WEB

## Quick start
1. Change the mem_limit in the [docker-compose.yaml](./docker-compose.yaml)
2. Change the  workercount in the [./gunicorn_conf.py](./gunicorn_conf.py)
3. Build the docker image with `docker-compose build`
4. Start the service with `docker-compose up`