# -*- coding: utf-8 -*-
"""
This file contains the constants that control model parameters.

Note: Please don't add keys directly here, refer to environment variables
"""
import os


# GRU
BATCH_SIZE_PROB_SEC_GRU = os.environ.get("BATCH_SIZE_PROB_SEC_GRU", 1024)
BATCH_SIZE_PROB_CVE_GRU = os.environ.get("BATCH_SIZE_PROB_GRU", 1024)

# BERT
BATCH_SIZE_PROB_SEC_BERT = os.environ.get("BATCH_SIZE_PROB_SEC_BERT", 64)
BATCH_SIZE_PROB_CVE_BERT = os.environ.get("BATCH_SIZE_PROB_CVE_BERT", 64)

# Transformers-pytorch-bert
TOKENIZER_CONVERT_LOWER_CASE = True
TASK_NAME = "sst-2"
MAX_SEQ_LENGTH = 512
GUID_PREFIX_INFERENCE = "dev"
BATCH_SIZE_PROB_CVE_BERT_TORCH = 64
