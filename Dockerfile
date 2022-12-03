FROM python:3.9
RUN apt-get update
#musl-dev
RUN apt-get install -y libffi-dev
RUN apt-get install -y make automake cmake gcc g++ subversion python3-dev
RUN apt-get install -y meson
RUN apt-get install -y libsndfile1
ENV STATIC_URL /static
ENV STATIC_PATH /var/www/app/static
COPY ./requirements.txt /var/www/requirements.txt
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r /var/www/requirements.txt
COPY ./ /app
WORKDIR /app
RUN python -m nltk.downloader averaged_perceptron_tagger
RUN python -m nltk.downloader cmudict
CMD ["gunicorn", "--conf", "gunicorn_conf.py", "--timeout", "0", "--bind", "0.0.0.0:80", "wsgi:app"]