"""Contains file utilities to read/write from S3 and local file system."""
import logging
import os

import daiquiri

from utils import cloud_constants as cc

daiquiri.setup(level=logging.DEBUG)
_logger = daiquiri.getLogger(__name__)


def write_output_csv(start_time, end_time, cve_model_type, ecosystem, df,
                     s3_upload):
    """Write the generated CSV to disk/S3."""
    # ======= PREPARING PROBABLE SECURITY & CVE DATASETS ========
    _logger.info("----- PREPARING PROBABLE SECURITY & CVE DATASETS  -----")

    base_triage_dir = os.environ.get("BASE_TRIAGE_DIR",
                                     "/data_assets/triaged_datasets")
    new_triage_subdir = "-".join([start_time.format("YYYYMMDD"),
                                  end_time.format("YYYYMMDD")])
    new_triage_results_dir = os.path.join(base_triage_dir, new_triage_subdir)
    if cve_model_type == "gru":
        file_prefix = "gru_model_inference_"
    elif cve_model_type == "bert":
        file_prefix = "bert_model_inference_"
    else:
        _logger.info("No Valid model type specified, defaulting to BERT model.")  # noqa
        file_prefix = "bert_model_inference_"
    model_inference_dataset_filename = "{file_prefix}full_output_{new_triage_subdir}_{ecosystem}.csv".format(  # noqa
        file_prefix=file_prefix, new_triage_subdir=new_triage_subdir, ecosystem=ecosystem  # noqa
    )
    model_inference_dataset = os.path.join(
        new_triage_results_dir, model_inference_dataset_filename,
    )
    probable_sec_cve_dataset_filename = (
        "{file_prefix}probable_security_and_cves_"
        "{new_triage_subdir}_{ecosystem}.csv".format(
            file_prefix=file_prefix, new_triage_subdir=new_triage_subdir,
            ecosystem=ecosystem
        )
    )

    probable_sec_cve_dataset = os.path.join(
        new_triage_results_dir, probable_sec_cve_dataset_filename
    )
    probable_cve_dataset_filename = "{file_prefix}probable_cves_{new_triage_subdir}_{ecosystem}.csv".format(  # noqa
        file_prefix=file_prefix, new_triage_subdir=new_triage_subdir,
        ecosystem=ecosystem
    )

    probable_cve_dataset = os.path.join(new_triage_results_dir,
                                        probable_cve_dataset_filename)
    if not s3_upload:
        if not os.path.exists(new_triage_results_dir):
            _logger.info(
                "Creating New Model Inference Directory: {}".format(new_triage_results_dir)  # noqa
            )
            os.makedirs(new_triage_results_dir)
        else:
            _logger.info(
                "Using Existing Model Inference Directory: {}".format(new_triage_results_dir)  # noqa
            )

    df.drop(["norm_description", "description"], inplace=True,
            errors="ignore", axis=1)
    df["triage_is_security"] = 0
    df["triage_is_cve"] = 0
    df["triage_feedback_comments"] = ""
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
    ]
    df = df[columns]
    df.loc[:, "ecosystem"] = ecosystem
    if not s3_upload:
        _logger.info("Saving Model Inference datasets locally: {}".format(model_inference_dataset))  # noqa
        df.to_csv(model_inference_dataset, index=False)
    else:
        s3_path = "s3://{bucket_name}/triaged_datasets/{triage_dir}/{model_inference_dataset_filename}".format(  # noqa
            bucket_name=cc.S3_BUCKET_NAME_INFERENCE,
            triage_dir=new_triage_subdir,
            model_inference_dataset_filename=model_inference_dataset_filename,
        )
        df.to_csv(
            s3_path, index=False,
        )
        _logger.info("Saving Model Inference dataset to S3: {}".format(s3_path))  # noqa

    # Now save the probable securities dataset.
    df = df[df.security_model_flag == 1].drop(["triage_is_cve"], axis=1)
    if not s3_upload:
        _logger.info("Saving Probable Security dataset locally:{}".format(probable_sec_cve_dataset))  # noqa
        df.to_csv(probable_sec_cve_dataset, index=False)
    else:
        s3_path = "s3://{bucket_name}/triaged_datasets/{triage_dir}/{probable_sec_cve_dataset_filename}".format(  # noqa
            bucket_name=cc.S3_BUCKET_NAME_INFERENCE,
            triage_dir=new_triage_subdir,
            probable_sec_cve_dataset_filename=probable_sec_cve_dataset_filename,  # noqa
        )
        df.to_csv(
            s3_path, index=False,
        )
        _logger.info("Saving probable security CVE dataset to S3: {}".format(s3_path))  # noqa

    # Now save the probable CVE dataset.
    df = df[df.cve_model_flag == 1].drop(["triage_is_security"], axis=1)
    if not s3_upload:
        _logger.info("Saving Probable CVE dataset locally:{}".format(probable_cve_dataset))  # noqa
        df.to_csv(probable_cve_dataset, index=False)
    else:
        s3_path = "s3://{bucket_name}/triaged_datasets/{triage_dir}/{probable_cve_dataset_filename}".format(  # noqa
            bucket_name=cc.S3_BUCKET_NAME_INFERENCE,
            triage_dir=new_triage_subdir,
            probable_cve_dataset_filename=probable_cve_dataset_filename,
        )
        df.to_csv(
            s3_path, index=False,
        )
        _logger.info("Saving Probable dataset to S3: {}".format(s3_path))
    return df
