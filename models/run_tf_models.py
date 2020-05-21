# -*- coding: utf-8 -*-
import gc
import logging

import daiquiri
import numpy as np
import tensorflow as tf
import pandas as pd

from models import cve_dl_classifier as cdc
from models import bert_cve_classifier as bcvec
from models import security_dl_classifier as sdc
from utils import bert_text_processor as btp
from utils import cloud_constants as cc
from utils import model_constants as mc
from utils import text_normalizer as tn

daiquiri.setup(level=logging.INFO)
_logger = daiquiri.getLogger(__name__)


def run_tensorflow_security_classifier(df: pd.DataFrame):
    """Run inference against the security issues classifier."""
    _logger.info("Text Pre-processing Issue/PR Descriptions")
    df["norm_description"] = tn.pre_process_documents_parallel(documents=df["description"].values)

    _logger.info("Setting Default CVE and Security Flags")
    df["security_model_flag"] = 0
    df["cve_model_flag"] = 0

    _logger.info("\n")
    _logger.info("----- STARTING MODEL INFERENCE -----")
    _logger.info("Loading Security Model")
    sc = sdc.SecurityClassifier(
        embedding_size=300,
        max_length=1000,
        max_features=800000,
        tokenizer_path=cc.P1GRU_SEC_MODEL_TOKENIZER_PATH,
        model_weights_path=cc.P1GRU_SEC_MODEL_WEIGHTS_PATH,
    )
    sc.build_model_architecture()
    sc.load_model_weights()
    sc_model = sc.get_model()

    _logger.info("Preparing data for security model inference")
    security_encoded_docs = sc.prepare_inference_data(df["norm_description"].tolist())
    _logger.info("Total Security Docs Encoded: {n}".format(n=len(security_encoded_docs)))
    sec_doc_lengths = np.array([len(np.nonzero(item)[0]) for item in security_encoded_docs])
    _logger.info("Removing bad docs with low tokens")
    sec_doc_idx = np.argwhere(sec_doc_lengths >= 5).ravel()
    filtered_security_encoded_docs = security_encoded_docs[sec_doc_idx]
    _logger.info(
        "Filtered Security Docs Encoded: {n}".format(n=len(filtered_security_encoded_docs))
    )
    _logger.info("Issue count before security issue filter: {c}".format(c=df.shape[0]))
    _logger.info("Making predictions for probable security issues")
    sec_pred_probs = sc_model.predict(
        filtered_security_encoded_docs, batch_size=mc.BATCH_SIZE_PROB_SEC_BERT, verbose=0
    )
    sec_pred_probsr = sec_pred_probs.ravel()
    sec_pred_labels = [
        1 if prob > mc.GRU_SEC_MODEL_PROB_THRESHOLD else 0 for prob in sec_pred_probsr
    ]
    _logger.info("Updating Security Model predictions in dataset")
    df.loc[df.index.isin(sec_doc_idx), "security_model_flag"] = sec_pred_labels

    _logger.info("Teardown security model")
    del sc
    del sc_model
    gc.collect()
    return df


def run_bert_tensorflow_model(df: pd.DataFrame):
    """Run CVE classifier using the bert-tensorflow BERT model."""
    df = run_tensorflow_security_classifier(df)
    _logger.info("\n")

    _logger.info("Keeping track of probable security issue rows")
    subset_df = df[df["security_model_flag"] == 1]
    prob_security_df_rowidx = np.array(subset_df.index)

    _logger.info("Issue count after security issue filter: {c}".format(c=subset_df.shape[0]))

    sess = tf.Session()
    BERT_MAX_SEQ_LEN = 512

    _logger.info("Loading CVE Model")
    bc = bcvec.BERTClassifier(
        bert_model_path=cc.BASE_BERT_UNCASED_PATH, max_seq_length=BERT_MAX_SEQ_LEN
    )
    bc.build_model_architecture()

    subset_df["norm_description"] = tn.pre_process_documents_parallel_bert(
        documents=subset_df["description"].values
    )
    cve_encoded_docs = subset_df["norm_description"].values
    _logger.info("Total CVE Docs Encoded: {n}".format(n=len(cve_encoded_docs)))
    cve_doc_lengths = np.array([len(doc.split(" ")) for doc in cve_encoded_docs])
    _logger.info("Removing bad docs with low tokens")
    cve_doc_idx = np.argwhere(cve_doc_lengths >= 10).ravel()
    filtered_cve_encoded_docs = cve_encoded_docs[cve_doc_idx]
    _logger.info("Filtered CVE Docs Encoded: {n}".format(n=len(filtered_cve_encoded_docs)))

    _logger.info("BERT text processing and feature engineering")
    btp_obj = btp.BertTextProcessor(
        tf_session=sess, bert_model_path=cc.BASE_BERT_UNCASED_PATH, max_seq_length=BERT_MAX_SEQ_LEN
    )
    btp_obj.create_bert_tokenizer()
    btp_obj.convert_text_to_input_examples(filtered_cve_encoded_docs)
    btp_obj.convert_examples_to_features()

    _logger.info("Making predictions for probable CVE issues")
    btp.initialize_vars(sess)
    bc.load_model_weights(model_weights_path=cc.P2BERT_CVE_MODEL_WEIGHTS_PATH)

    cve_pred_probs = bc.model_estimator.predict(
        x=[btp_obj.input_ids, btp_obj.input_masks, btp_obj.segment_ids],
        batch_size=mc.BATCH_SIZE_PROB_CVE_BERT,
        verbose=1,
    )
    cve_pred_probsr = cve_pred_probs.ravel()
    cve_pred_labels = [
        1 if prob > mc.BERT_CVE_PROB_THRESHOLD_LEGACY else 0 for prob in cve_pred_probsr
    ]
    from collections import Counter

    _logger.info(Counter(cve_pred_labels))
    _logger.info("Updating CVE Model predictions in dataset")
    prob_cve_idxs = prob_security_df_rowidx[cve_doc_idx]
    df.loc[df.index.isin(prob_cve_idxs), "cve_model_flag"] = cve_pred_labels
    _logger.info("Teardown CVE model")
    _logger.info("\n")

    del btp_obj
    del bc
    gc.collect()
    return df


def run_gru_cve_model(df):
    """Run inference against the GRU based CVE classifier model."""
    df = run_tensorflow_security_classifier(df)
    _logger.info("Loading CVE Model")
    cvc = cdc.CVEClassifier(
        embedding_size=300,
        max_length=1000,
        max_features=600000,
        tokenizer_path=cc.P1GRU_CVE_MODEL_TOKENIZER_PATH,
        model_weights_path=cc.P1GRU_CVE_MODEL_WEIGHTS_PATH,
    )
    cvc.build_model_architecture()
    cvc.load_model_weights()
    cc_model = cvc.get_model()

    _logger.info("Keeping track of probable security issue rows")
    subset_df = df[df["security_model_flag"] == 1]
    prob_security_df_rowidx = np.array(subset_df.index)

    cve_encoded_docs = cvc.prepare_inference_data(subset_df["norm_description"].tolist())
    _logger.info("Total CVE Docs Encoded: {n}".format(n=len(cve_encoded_docs)))
    cve_doc_lengths = np.array([len(np.nonzero(item)[0]) for item in cve_encoded_docs])
    _logger.info("Removing bad docs with low tokens")
    cve_doc_idx = np.argwhere(cve_doc_lengths >= 10).ravel()
    filtered_cve_encoded_docs = cve_encoded_docs[cve_doc_idx]
    _logger.info("Filtered CVE Docs Encoded: {n}".format(n=len(filtered_cve_encoded_docs)))

    _logger.info("Making predictions for probable CVE issues")
    cve_pred_probs = cc_model.predict(
        filtered_cve_encoded_docs, batch_size=mc.BATCH_SIZE_PROB_CVE_GRU, verbose=0
    )
    cve_pred_probsr = cve_pred_probs.ravel()
    cve_pred_labels = [1 if prob > 0.3 else 0 for prob in cve_pred_probsr]
    _logger.info("Updating CVE Model predictions in dataset")
    prob_cve_idxs = prob_security_df_rowidx[cve_doc_idx]
    df.loc[df.index.isin(prob_cve_idxs), "cve_model_flag"] = cve_pred_labels
    _logger.info("Teardown CVE model")
    _logger.info("\n")

    del cvc
    del cc_model
    gc.collect()
    return df
