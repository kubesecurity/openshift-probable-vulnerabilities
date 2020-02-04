# -*- coding: utf-8 -*-
"""
This file contains the constants for interaction with AWS/Minio.
Note: Please don't add keys directly here, refer to environment variables
"""

import os

# Please make sure you have your AWS envt variables setup
AWS_S3_REGION = os.environ.get('AWS_S3_REGION', 'ap-south-1')
AWS_S3_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID', '')
AWS_S3_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY', '')
S3_BUCKET_NAME = os.environ.get('AWS_S3_BUCKET_NAME', 'avgupta-dev-gokube-triage')

# Please set the following to point to your BQ auth credentials JSON
BIGQUERY_CREDENTIALS_FILEPATH = os.environ.get('BIGQUERY_CREDENTIALS_FILEPATH', '../../auth/bq_key.json')

GOKUBE_REPO_LIST = './utils/data_assets/golang-repo-list.txt'
KNATIVE_REPO_LIST = './utils/data_assets/knative-repo-list.txt'
KUBEVIRT_REPO_LIST = './utils/data_assets/kubevirt-repo-list.txt'

P1GRU_SEC_MODEL_TOKENIZER_PATH = './models/model_assets/gokube-phase1-jun19/embeddings/security_tokenizer_word2idx_fulldata.pkl'
P1GRU_SEC_MODEL_WEIGHTS_PATH = './models/model_assets/gokube-phase1-jun19/saved_models/security_model_train99-jun19_weights.h5'

P1GRU_CVE_MODEL_TOKENIZER_PATH = './models/model_assets/gokube-phase1-jun19/embeddings/cve_tokenizer_word2idx_fulldata.pkl'
P1GRU_CVE_MODEL_WEIGHTS_PATH = './models/model_assets/gokube-phase1-jun19/saved_models/cve_model_train99-jun19_weights.h5'

BASE_BERT_UNCASED_PATH = './models/model_assets/gokube-phase2/base_bert_tfhub_models/bert_uncased_L12_H768_A12'
P2BERT_CVE_MODEL_WEIGHTS_PATH = './models/model_assets/gokube-phase2/saved_models/bert_cve75_weights-ep:02-trn_loss:0.172-trn_acc:0.957-val_loss:0.164-val_acc:0.978.h5'
