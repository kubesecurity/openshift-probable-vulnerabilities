"""Wrapper module to run inference using the transformers library based torch impl. of BERT CVE classifier."""
import logging

import daiquiri
import pandas as pd

import utils.cloud_constants as cc
from models.bert_cve_classifier_torch import BertTorchCVEClassifier
from utils import text_normalizer as tn

daiquiri.setup(level=logging.INFO)
_logger = daiquiri.getLogger(__name__)


def run_torch_cve_model_bert(
    df: pd.DataFrame, custom_model_path=None, batch_size_prediction=None
) -> pd.DataFrame:
    """Run inference on the supplied dataframe and return the same with `cve_model` column filled."""
    _logger.info("Starting inference on security issues.")
    _logger.info("Keeping track of probable security issue rows")
    subset_df = df[df["security_model_flag"] == 1]

    subset_df["norm_description"] = tn.pre_process_documents_parallel_bert(
        documents=subset_df["description"].values
    )
    subset_df = subset_df[subset_df.norm_description.apply(lambda x: len(str(x).split()) > 10)]
    _logger.info(
        "Issue count after security issue filter and bad sample elimination: {c}".format(
            c=subset_df.shape[0]
        )
    )
    clf = BertTorchCVEClassifier(custom_model_path or cc.P2_PYTORCH_CVE_BERT_CLASSIFIER_PATH)
    predictions = clf.predict(subset_df, batch_size_prediction)
    df.loc[df.index.isin(subset_df.index), "cve_model_flag"] = predictions
    return df
