# TTS-WEB

## Quick start
1. Change the mem_limit in the [docker-compose.yaml](./docker-compose.yaml)
2. Change the  workercount in the [./gunicorn_conf.py](./gunicorn_conf.py)
3. Add the pytorch_model.bin to the [./de_to_ch/experiments/transcribed_version__20220721_104626/best-model/](./de_to_ch/experiments/transcribed_version__20220721_104626/best-model/) folder.
4. Build the docker image with `docker-compose build`
5. Start the service with `docker-compose up`