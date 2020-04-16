import logging

import daiquiri
import pandas

import utils.cloud_constants as cc
from models.bert_cve_classifier_torch import BertTorchCVEClassifier
from utils import text_normalizer as tn

daiquiri.setup(level=logging.INFO)
_logger = daiquiri.getLogger(__name__)


def run_torch_cve_model_bert(df: pandas.DataFrame) -> pandas.DataFrame:
    _logger.info("Starting inference on security issues.")
    _logger.info("Keeping track of probable security issue rows")
    subset_df = df[df["security_model_flag"] == 1]

    subset_df["norm_description"] = tn.pre_process_documents_parallel_bert(
        documents=subset_df["description"].values
    )
    _logger.info("Issue count after security issue filter: {c}".format(c=subset_df.shape[0]))
    clf = BertTorchCVEClassifier(cc.P2_PYTORCH_CVE_BERT_CLASSIFIER_PATH)
    predictions = clf.predict(subset_df)
    df.loc[df.index.isin(subset_df.index), "cve_model_flag"] = predictions
    return df
