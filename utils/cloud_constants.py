# -*- coding: utf-8 -*-
"""Contains the constants for interaction with AWS/Minio."""

import os
from s3fs import S3FileSystem

# Please make sure you have your AWS envt variables setup
AWS_S3_REGION = os.environ.get("AWS_S3_REGION", "ap-south-1")
S3_MODEL_ACCESS_KEY_ID = os.environ.get("S3_MODEL_ACCESS_KEY_ID", "")
S3_MODEL_SECRET_ACCESS_KEY = os.environ.get("S3_MODEL_SECRET_ACCESS_KEY", "")
S3_INFERENCE_ACCESS_KEY_ID = os.environ.get("S3_INFERENCE_ACCESS_KEY_ID", "")
S3_INFERENCE_SECRET_ACCESS_KEY = os.environ.get("S3_INFERENCE_SECRET_ACCESS_KEY", "")
S3_BUCKET_NAME_MODEL = os.environ.get("S3_MODEL_BUCKET", "avgupta-dev-gokube-triage")
S3_BUCKET_NAME_INFERENCE = os.environ.get("S3_INFERENCE_BUCKET", "avgupta-dev-gokube-triage")

# Please set the following to point to your BQ auth credentials JSON
BIGQUERY_CREDENTIALS_FILEPATH = os.environ.get(
    "BIGQUERY_CREDENTIALS_FILEPATH", "../../auth/bq_key.json"
)

GOKUBE_REPO_LIST = os.environ.get("GOKUBE_REPO_LIST", "/utils/data_assets/golang-repo-list.txt")
KNATIVE_REPO_LIST = os.environ.get("KNATIVE_REPO_LIST", "/utils/data_assets/knative-repo-list.txt")
KUBEVIRT_REPO_LIST = os.environ.get(
    "KUBEVIRT_REPO_LIST", "/utils/data_assets/kubevirt-repo-list.txt"
)

P1GRU_SEC_MODEL_TOKENIZER_PATH = os.environ.get(
    "P1GRU_SEC_MODEL_TOKENIZER_PATH",
    "/model_assets/gokube-phase1-jun19/embeddings/security_tokenizer_word2idx_fulldata.pkl",
)
P1GRU_SEC_MODEL_WEIGHTS_PATH = os.environ.get(
    "P1GRU_SEC_MODEL_WEIGHTS_PATH",
    "/model_assets/gokube-phase1-jun19/saved_models/security_model_train99-jun19_weights.h5",
)

P1GRU_CVE_MODEL_TOKENIZER_PATH = os.environ.get(
    "P1GRU_CVE_MODEL_TOKENIZER_PATH",
    "/model_assets/gokube-phase1-jun19/embeddings/cve_tokenizer_word2idx_fulldata.pkl",
)
P1GRU_CVE_MODEL_WEIGHTS_PATH = os.environ.get(
    "P1GRU_CVE_MODEL_WEIGHTS_PATH",
    "/model_assets/gokube-phase1-jun19/saved_models/cve_model_train99-jun19_weights.h5",
)

BASE_BERT_UNCASED_PATH = os.environ.get(
    "BASE_BERT_UNCASED_PATH",
    "/model_assets/gokube-phase2/base_bert_tfhub_models/bert_uncased_L12_H768_A12",
)
P2BERT_CVE_MODEL_WEIGHTS_PATH = os.environ.get(
    "P2BERT_CVE_MODEL_WEIGHTS_PATH",
    "/model_assets/gokube-phase2/saved_models/bert_cve75_weights-ep:02-trn_loss:0.172-trn_acc:0.957-val_loss:0.164-val_acc:0.978.h5",  # noqa
)

P2_PYTORCH_CVE_BERT_CLASSIFIER_PATH = os.environ.get(
    "P2_PYTORCH_CVE_BERT_CLASSIFIER_PATH",
    "/model_assets/gokube-phase2/pytorch-cve-warmup-2020_06_02_13_48_17/",
)

S3_MODEL_REFRESH = os.environ.get("S3_MODEL_REFRESH", "True")

# TODO: Use this constant later to not download everything to disk, leave it for now disk is not a problem.
MODEL_ASSETS = {
    "sec_model": [
        P1GRU_SEC_MODEL_TOKENIZER_PATH.lstrip("/"),
        P1GRU_SEC_MODEL_WEIGHTS_PATH.lstrip("/"),
    ],
    "gru": [P1GRU_CVE_MODEL_TOKENIZER_PATH.lstrip("/"), P1GRU_CVE_MODEL_WEIGHTS_PATH.lstrip("/")],
    "bert": [P2BERT_CVE_MODEL_WEIGHTS_PATH.lstrip("/"), BASE_BERT_UNCASED_PATH.lstrip("/")],
    "bert_torch": [P2_PYTORCH_CVE_BERT_CLASSIFIER_PATH.lstrip("/")],
}

INFERENCE_DROP_DESCRIPTIONS = os.environ.get("INFERENCE_DROP_DESCRIPTIONS", "True")

SKIP_INSERT_API_CALL = os.environ.get("SKIP_INSERT_API_CALL", "false").lower() == "true"
BASE_TRIAGE_DIR = os.environ.get("BASE_TRIAGE_DIR", "/data_assets/triaged_datasets")
NEW_TRIAGE_SUBDIR = "{stat_time}-{end_time}"
S3_FILE_PATH = "s3://{bucket_name}/triaged_datasets/{triage_dir}/{dataset_filename}"
OUTPUT_FILE_NAME = "{file_prefix}_{data_type}_{triage_dir}_{ecosystem}.csv"
FULL_OUTPUT = "full_output"
PROBABLE_SECURITY_AND_CVES = "probable_security_and_cves"
PROBABLE_CVES = "probable_cves"
FAILED_TO_INSERT_DATA_RAW_PATH = "s3://{bucket_name}/triaged_datasets/{triage_dir}/{failed_dataset_filename}"
FAILED_TO_INSERT = "failed_to_insert"

OSA_API_SERVER_HOST = os.environ.get("OSA_API_SERVER_HOST", "osa-api-server")
OSA_API_SERVER_PORT = os.environ.get("OSA_API_SERVER_PORT", 5000)
OSA_API_SERVER_URL = 'http://{host}:{port}'.format(host=OSA_API_SERVER_HOST, port=OSA_API_SERVER_PORT)
DATA_INSERT_CONCURRENCY = int(os.environ.get("DATA_INSERT_CONCURRENCY", 10))
# Shared instance of S3FileSystem
INFERENCE_S3FS = S3FileSystem(key=S3_INFERENCE_ACCESS_KEY_ID, secret=S3_INFERENCE_SECRET_ACCESS_KEY)
