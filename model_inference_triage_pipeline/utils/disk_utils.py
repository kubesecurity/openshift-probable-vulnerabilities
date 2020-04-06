import logging
import os

import daiquiri

daiquiri.setup(level=logging.DEBUG)
_logger = daiquiri.getLogger(__name__)


def write_output_csv_disk(start_time, end_time, cve_model_type, ecosystem, df):
    """Write the generated CSV to disk."""
    # ======= PREPARING PROBABLE SECURITY & CVE DATASETS ========
    _logger.info("----- PREPARING PROBABLE SECURITY & CVE DATASETS  -----")

    base_triage_dir = os.environ.get("BASE_TRIAGE_DIR", "/data_assets/triaged_datasets")
    new_triage_subdir = "-".join(
        [start_time.format("YYYYMMDD"), end_time.format("YYYYMMDD")]
    )
    new_triage_results_dir = os.path.join(base_triage_dir, new_triage_subdir)
    if cve_model_type == "gru":
        file_prefix = "gru_model_inference_"
    elif cve_model_type == "bert":
        file_prefix = "bert_model_inference_"
    else:
        _logger.info("No Valid model type specified, defaulting to BERT model.")
        file_prefix = "bert_model_inference_"
    model_inference_dataset = os.path.join(
        new_triage_results_dir,
        file_prefix + "full_output_" + new_triage_subdir + "_" + ecosystem + ".csv",
    )
    probable_sec_cve_dataset = os.path.join(
        new_triage_results_dir,
        file_prefix
        + "probable_security_and_cves_"
        + new_triage_subdir
        + "_"
        + ecosystem
        + ".csv",
    )
    probable_cve_dataset = os.path.join(
        new_triage_results_dir,
        file_prefix + "probable_cves_" + new_triage_subdir + "_" + ecosystem + ".csv",
    )
    if not os.path.exists(new_triage_results_dir):
        _logger.info(
            "Creating New Model Inference Directory: {}".format(new_triage_results_dir)
        )
        os.makedirs(new_triage_results_dir)
    else:
        _logger.info(
            "Using Existing Model Inference Directory: {}".format(
                new_triage_results_dir
            )
        )

    df.drop(["norm_description", "description"], inplace=True, errors="ignore", axis=1)
    df["triage_is_security"] = 0
    df["triage_is_cve"] = 0
    df["triage_feedback_comments"] = ""
    columns = [
        "ecosystem",
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
    _logger.info(
        "Saving Model Inference datasets locally:".format(model_inference_dataset)
    )
    df.to_csv(model_inference_dataset, index=False)
    _logger.info("Saving Probable Security dataset:{}".format(probable_sec_cve_dataset))
    df[df.security_model_flag == 1].drop(["triage_is_cve"], axis=1).to_csv(
        probable_sec_cve_dataset, index=False
    )
    _logger.info("Saving Probable CVE dataset: {}".format(probable_cve_dataset))
    df[df.cve_model_flag == 1].drop(["triage_is_security"], axis=1).to_csv(
        probable_cve_dataset, index=False
    )
    return new_triage_results_dir
