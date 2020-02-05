# Based on the RHEL-8 UBI
FROM docker.io/nvidia/cuda:10.1-devel-ubi8
LABEL MAINTAINER="Avishkar Gupta <avgupta@redhat.com>"

RUN dnf install -y git python3 && python3 -m pip install --upgrade pip virtualenv setuptools

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Helps speedup docker build time.
ADD model_inference_triage_pipeline/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

ADD model_inference_triage_pipeline /model_inference_triage_pipeline
ADD model_inference_triage_pipeline/utils/data_assets/ /data_assets/
ADD deployment/entrypoint.sh /entrypoint.sh

RUN mkdir /model_assets/ && chmod +x /entrypoint.sh && chmod -R 0777 /data_assets/ /model_assets/

ENTRYPOINT ["/entrypoint.sh"]
