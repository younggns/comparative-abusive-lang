FROM python:3.6.12-slim

USER root
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt | pip install -r /dev/stdin
