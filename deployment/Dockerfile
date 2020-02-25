# Based on the RHEL-7 UBI
FROM docker.io/nvidia/cuda:10.0-cudnn7-devel-ubi7
LABEL MAINTAINER="Avishkar Gupta <avgupta@redhat.com>"

RUN yum install -y git rh-python36 \
    && source /opt/rh/rh-python36/enable \
    && python -m pip install --upgrade pip virtualenv setuptools

ENV VIRTUAL_ENV=/opt/venv
RUN source /opt/rh/rh-python36/enable && python -m virtualenv --python=$(which python) $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Helps speedup docker build time.
ADD model_inference_triage_pipeline/requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

ADD model_inference_triage_pipeline /model_inference_triage_pipeline
ADD model_inference_triage_pipeline/utils/data_assets/ /data_assets/
ADD deployment/entrypoint.sh /entrypoint.sh

RUN mkdir /model_assets/ && chmod +x /entrypoint.sh && chmod -R 0777 /data_assets/ /model_assets/

ENTRYPOINT ["/entrypoint.sh"]
