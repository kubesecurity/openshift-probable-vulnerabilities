# Based on the RHEL-8 UBI
FROM docker.io/nvidia/cuda:10.1-devel-ubi8
LABEL MAINTAINER="Avishkar Gupta <avgupta@redhat.com>"

RUN dnf install -y git python3 && python3 -m pip install --upgrade pip virtualenv

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ADD model_inference_triage_pipeline /model_inference_triage_pipeline
ADD model_inference_triage_pipeline/utils/data_assets/ /data_assets/
ADD deployment/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN chmod -R 0777 /data_assets/ /model_assets/
RUN pip install -r /model_inference_triage_pipeline/requirements.txt

ENTRYPOINT ["/entrypoint.sh"]
