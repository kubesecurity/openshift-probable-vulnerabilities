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
BATCH_SIZE_PROB_SEC_BERT = os.environ.get("BATCH_SIZE_PROB_SEC_BERT", 32)
BATCH_SIZE_PROB_CVE_BERT = os.environ.get("BATCH_SIZE_PROB_CVE_BERT", 32)
