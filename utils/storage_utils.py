"""Contains file utilities to read/write from S3 and local file system."""
import logging
import os

import daiquiri

from utils import cloud_constants as cc
from utils import other_constants as oc

daiquiri.setup(level=logging.DEBUG)
_logger = daiquiri.getLogger(__name__)


def write_output_csv(start_time, end_time, cve_model_type, ecosystem, df, s3_upload):
    """Write the generated CSV to disk/S3."""
    # ======= PREPARING PROBABLE SECURITY & CVE DATASETS ========
    _logger.info("----- PREPARING PROBABLE SECURITY & CVE DATASETS  -----")

    new_triage_subdir = oc.NEW_TRIAGE_SUBDIR.format(stat_time=start_time.format("YYYYMMDD"),
                                                    end_time=end_time.format("YYYYMMDD"))
    new_triage_results_dir = os.path.join(oc.BASE_TRIAGE_DIR, new_triage_subdir)
    file_prefix = get_file_prefix(cve_model_type)

    if not s3_upload:
        if not os.path.exists(new_triage_results_dir):
            _logger.info(
                "Creating New Model Inference Directory: {}".format(new_triage_results_dir)
            )
            os.makedirs(new_triage_results_dir)
        else:
            _logger.info(
                "Using Existing Model Inference Directory: {}".format(new_triage_results_dir)
            )

    df.drop(["norm_description", "description"], inplace=True, errors="ignore", axis=1)
    df["triage_is_security"] = 0
    df["triage_is_cve"] = 0
    df["triage_feedback_comments"] = ""

    df.loc[:, "ecosystem"] = ecosystem
    df.loc[:, "body"] = df.apply(lambda x: _restrict_data_length(x['body']), axis=1)
    columns = [
        "repo_name",
        "event_type",
        "status",
        "url",
        "security_model_flag",
        "cve_model_flag",
        "triage_is_security",
        "triage_is_cve",
        "triage_feedback_comments",
        "id",
        "number",
        "api_url",
        "created_at",
        "updated_at",
        "closed_at",
        "creator_name",
        "creator_url",
        "ecosystem",
        "title",
        "body"
    ]
    df = df[columns]
    save_data_to_csv(df, s3_upload, file_prefix, new_triage_subdir, ecosystem, oc.FULL_OUTPUT)

    # Now save the probable securities dataset.
    df = df[df.security_model_flag == 1].drop(["triage_is_cve"], axis=1)
    save_data_to_csv(df, s3_upload, file_prefix, new_triage_subdir, ecosystem, oc.PROBABLE_SECURITY_AND_CVES)

    # Now save the probable CVE dataset.
    df = df[df.cve_model_flag == 1].drop(["triage_is_security"], axis=1)
    save_data_to_csv(df, s3_upload, file_prefix, new_triage_subdir, ecosystem, oc.PROBABLE_CVES)

    return df


def _restrict_data_length(data: str) -> str:
    """Ristrict long length data content."""
    return data[0:oc.MAX_STRING_LEN_FOR_CSV_EXPORT] if data else data


def get_file_prefix(cve_model_type: str) -> str:
    """Get prefix based on cve_model_type."""
    if cve_model_type == "gru":
        return "gru_model_inference"
    elif cve_model_type == "bert":
        return "bert_model_inference"
    else:
        _logger.info("No Valid model type specified, defaulting to BERT model.")
        return "bert_model_inference"


def save_data_to_csv(df, s3_upload, file_prefix, new_triage_subdir, ecosystem, data_type):
    """Save dataframe data to s3/local file system."""
    new_triage_results_dir = os.path.join(oc.BASE_TRIAGE_DIR, new_triage_subdir)
    filename = oc.OUTPUT_FILE_NAME.format(file_prefix=file_prefix, data_type=data_type, triage_dir=new_triage_subdir,
                                          ecosystem=ecosystem)
    if not s3_upload:
        dataset = os.path.join(new_triage_results_dir, filename)
        _logger.info("Saving {} dataset locally:{}".format(data_type, dataset))
        df.to_csv(dataset, index=False)
    else:
        s3_path = cc.S3_FILE_PATH.format(bucket_name=cc.S3_BUCKET_NAME_INFERENCE, triage_dir=new_triage_subdir,
                                         dataset_filename=filename)
        with cc.INFERENCE_S3FS.open(s3_path, 'w') as f:
            df.to_csv(f, index=False)
        _logger.info("Saving {} dataset to {}".format(data_type, s3_path))
