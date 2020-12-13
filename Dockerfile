FROM python:3.6.12-slim
USER root
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install poetry
