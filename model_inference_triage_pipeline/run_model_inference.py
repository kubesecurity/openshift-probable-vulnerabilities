import argparse
import logging
import sys
import textwrap
import warnings

import arrow
import daiquiri
import pandas as pd
import torch

from utils import aws_utils as aws
from utils import bq_client_helper as bq_helper
from utils import cloud_constants as cc
from utils.disk_utils import write_output_csv_disk

daiquiri.setup(level=logging.INFO)
_logger = daiquiri.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)


def main():
    """The main program logic."""
    parser = get_argument_parser()
    args = parser.parse_args()

    DAYS_SINCE_YDAY = args.days_since_yday
    ECO_SYSTEM = args.eco_system.lower()
    COMPUTE_DEVICE = args.compute_device.lower()
    if COMPUTE_DEVICE != "cpu":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    SEC_MODEL_TYPE = args.security_filter_model.lower()
    if SEC_MODEL_TYPE != "" and SEC_MODEL_TYPE != "gru":
        _logger.error("GRU is the only supported security model type.")
        sys.exit(0)
    CVE_MODEL_TYPE = args.probable_cve_model.lower()
    S3_UPLOAD = args.s3_upload_cves

    day_count, start_time, end_time, date_range = setup_dates_for_triage(
        days_since_yday=DAYS_SINCE_YDAY
    )
    df = get_bq_data_for_inference(ECO_SYSTEM, day_count, date_range)
    df = run_inference(df, CVE_MODEL_TYPE)
    triage_results_dir = write_output_csv_disk(
        start_time, end_time, cve_model_type=CVE_MODEL_TYPE, ecosystem=ECO_SYSTEM, df=df
    )
    if S3_UPLOAD:
        upload_inference_results_s3(triage_results_dir)


def get_argument_parser():
    """Defines all the command line arguments for the program."""
    parser = argparse.ArgumentParser(
        prog="python",
        description=textwrap.dedent(
            """\
                        This script can be used to run our
                        AI models for probable vulnerability predictions.
                        Check usage patterns below"""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
             Usage patterns for inference script
             -----------------------------------
             The -days flag should be used with number of prev days data you want to pull
    
             1. GRU Models (Baseline): python run_model_inference.py -days=7 -device=gpu -sec-model=gru -cve-model=gru 
             2. BERT Model (CVE): python run_model_inference.py -days=7 -device=gpu -sec-model=gru -cve-model=bert 
             3. CPU inference: 
                    python run_model_inference.py -days=7 -device=cpu -sec-model=gru -cve-model=gru  
             """
        ),
    )

    parser.add_argument(
        "-days",
        "--days-since-yday",
        type=int,
        default=7,
        help="The number of days worth of data to retrieve from GitHub including yesterday",
    )

    parser.add_argument(
        "-eco",
        "--eco-system",
        type=str,
        default="openshift",
        choices=["openshift", "knative", "kubevirt"],
        help="The eco-system to monitor",
    )

    parser.add_argument(
        "-device",
        "--compute-device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu", "GPU", "CPU"],
        help="[Not implemented should work automatically] The backend device to run the AI models on - GPU or CPU",
    )

    parser.add_argument(
        "-sec-model",
        "--security-filter-model",
        type=str,
        default="gru",
        choices=["gru", "GRU"],
        help="The AI Model to use for security filtering - Model 1",
    )

    parser.add_argument(
        "-cve-model",
        "--probable-cve-model",
        type=str,
        default="gru",
        choices=["gru", "GRU", "bert", "BERT"],
        help="The AI Model to use for probable CVE predictions - Model 2",
    )

    parser.add_argument(
        "-s3-upload",
        "--s3-upload-cves",
        type=bool,
        default=False,
        choices=[False, True],
        help=(
            "Uploads inference CSVs to S3 bucket - turn off since bucket works only with my account for now (write"
            " access"
        ),
    )
    return parser


def setup_dates_for_triage(days_since_yday):
    """Sets up the date range for data retireval."""
    # ======= DATES SETUP FOR GETTING GITHUB BQ DATA ========
    _logger.info("----- DATES SETUP FOR GETTING GITHUB BQ DATA -----")

    # Don't change this
    present_time = arrow.now()

    # CHANGE NEEDED
    # to get data for N days back starting from YESTERDAY
    # e.g if today is 20190528 and DURATION DAYS = 2 -> BQ will get data for 20190527, 20190526
    # We don't get data for PRESENT DAY since github data will be incomplete on the same day
    # But you can get it if you want but better not to for completeness :)

    # You can set this directly from command line using the -d or --days-since-yday argument
    day_count = days_since_yday or 3  # Gets 3 days of previous data including YESTERDAY

    # Don't change this
    # Start time for getting data
    start_time = present_time.shift(days=-day_count)

    # Don't change this
    # End time for getting data (present_time - 1) i.e yesterday
    # you can remove -1 to get present day data
    # but it is not advised as data will be incomplete
    end_time = present_time.shift(days=-1)

    date_range = [dt.format("YYYYMMDD") for dt in arrow.Arrow.range("day", start_time, end_time)]
    _logger.info(
        "Data will be retrieved for Last N={n} days: {days}\n".format(n=day_count, days=date_range)
    )
    return day_count, start_time, end_time, date_range


def get_bq_data_for_inference(ecosystem, day_count, date_range) -> pd.DataFrame:
    """Query bigquery to retrieve data that is required for running the inference."""
    # ======= BQ CLIENT SETUP FOR GETTING GITHUB BQ DATA ========
    _logger.info("----- BQ CLIENT SETUP FOR GETTING GITHUB BQ DATA -----")

    GH_BQ_CLIENT = bq_helper.create_github_bq_client()
    if ecosystem == "openshift":
        REPO_NAMES = bq_helper.get_gokube_trackable_repos(repo_dir=cc.GOKUBE_REPO_LIST)
    elif ecosystem == "knative":
        REPO_NAMES = bq_helper.get_gokube_trackable_repos(repo_dir=cc.KNATIVE_REPO_LIST)
    elif ecosystem == "kubevirt":
        REPO_NAMES = bq_helper.get_gokube_trackable_repos(repo_dir=cc.KUBEVIRT_REPO_LIST)
    else:
        _logger.error(
            "Unsupported ecosystem, please re-run the inference with a valid ecosystem parameter."
        )
        # Returning 0 because this isn't a "HARD" error.
        sys.exit(0)

    setup_dates_for_triage(day_count)
    # ======= BQ QUERY PARAMS SETUP FOR GETTING GITHUB BQ DATA ========
    _logger.info("----- BQ QUERY PARAMS SETUP FOR GETTING GITHUB BQ DATA -----")

    # Don't change this
    YEAR_PREFIX = "20*"
    DAY_LIST = [item[2:] for item in date_range]
    QUERY_PARAMS = {
        "{year_prefix_wildcard}": YEAR_PREFIX,
        "{year_suffix_month_day}": "(" + ", ".join(["'" + d + "'" for d in DAY_LIST]) + ")",
        "{repo_names}": "(" + ", ".join(["'" + r + "'" for r in REPO_NAMES]) + ")",
    }

    _logger.info("\n")

    # ======= BQ GET DATASET SIZE ESTIMATE ========
    _logger.info("----- BQ Dataset Size Estimate -----")

    query = """
    SELECT  type as EventType, count(*) as Freq
            FROM `githubarchive.day.{year_prefix_wildcard}`
            WHERE _TABLE_SUFFIX IN {year_suffix_month_day}
            AND repo.name in {repo_names}
            AND type in ('PullRequestEvent', 'IssuesEvent')
            GROUP BY type
    """
    query = bq_helper.bq_add_query_params(query, QUERY_PARAMS)
    df = GH_BQ_CLIENT.query_to_pandas(query)
    _logger.info("Dataset Size for Last N={n} days:-".format(n=day_count))
    _logger.info("\n{data}".format(data=df))

    _logger.info("\n")

    # ======= BQ GITHUB DATASET RETRIEVAL & PROCESSING ========
    _logger.info("----- BQ GITHUB DATASET RETRIEVAL & PROCESSING -----")

    ISSUE_QUERY = """
    SELECT
        repo.name as repo_name,
        type as event_type,
        'golang' as ecosystem,
        JSON_EXTRACT_SCALAR(payload, '$.action') as status,
        JSON_EXTRACT_SCALAR(payload, '$.issue.id') as id,
        JSON_EXTRACT_SCALAR(payload, '$.issue.number') as number,
        JSON_EXTRACT_SCALAR(payload, '$.issue.url') as api_url,
        JSON_EXTRACT_SCALAR(payload, '$.issue.html_url') as url,
        JSON_EXTRACT_SCALAR(payload, '$.issue.user.login') as creator_name,
        JSON_EXTRACT_SCALAR(payload, '$.issue.user.html_url') as creator_url,
        JSON_EXTRACT_SCALAR(payload, '$.issue.created_at') as created_at,
        JSON_EXTRACT_SCALAR(payload, '$.issue.updated_at') as updated_at,
        JSON_EXTRACT_SCALAR(payload, '$.issue.closed_at') as closed_at,
        TRIM(REGEXP_REPLACE(
                 REGEXP_REPLACE(
                     JSON_EXTRACT_SCALAR(payload, '$.issue.title'),
                     r'\\r\\n|\\r|\\n',
                     ' '),
                 r'\s{2,}',
                 ' ')) as title,
        TRIM(REGEXP_REPLACE(
                 REGEXP_REPLACE(
                     JSON_EXTRACT_SCALAR(payload, '$.issue.body'),
                     r'\\r\\n|\\r|\\n',
                     ' '),
                 r'\s{2,}',
                 ' ')) as body
    
    FROM `githubarchive.day.{year_prefix_wildcard}`
        WHERE _TABLE_SUFFIX IN {year_suffix_month_day}
        AND repo.name in {repo_names}
        AND type = 'IssuesEvent'
        """

    ISSUE_QUERY = bq_helper.bq_add_query_params(ISSUE_QUERY, QUERY_PARAMS)
    qsize = GH_BQ_CLIENT.estimate_query_size(ISSUE_QUERY)
    _logger.info("Retrieving GH Issues. Query cost in GB={qc}".format(qc=qsize))

    issues_df = GH_BQ_CLIENT.query_to_pandas(ISSUE_QUERY)
    if issues_df.empty:
        _logger.warn("No issues present for given time duration.")
    else:
        _logger.info("Total issues retrieved: {n}".format(n=len(issues_df)))

        issues_df.created_at = pd.to_datetime(issues_df.created_at)
        issues_df.updated_at = pd.to_datetime(issues_df.updated_at)
        issues_df.closed_at = pd.to_datetime(issues_df.closed_at)
        issues_df = issues_df.loc[
            issues_df.groupby("url").updated_at.idxmax(skipna=False)
        ].reset_index(drop=True)
        _logger.info("Total issues after deduplication: {n}".format(n=len(issues_df)))

    PR_QUERY = """
    SELECT
        repo.name as repo_name,
        type as event_type,
        'golang' as ecosystem,
        JSON_EXTRACT_SCALAR(payload, '$.action') as status,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') as id,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') as number,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.url') as api_url,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') as url,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') as creator_name,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.html_url') as creator_url,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at') as created_at,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.updated_at') as updated_at,
        JSON_EXTRACT_SCALAR(payload, '$.pull_request.closed_at') as closed_at,
        TRIM(REGEXP_REPLACE(
                 REGEXP_REPLACE(
                     JSON_EXTRACT_SCALAR(payload, '$.pull_request.title'),
                     r'\\r\\n|\\r|\\n',
                     ' '),
                 r'\s{2,}',
                 ' ')) as title,
        TRIM(REGEXP_REPLACE(
                 REGEXP_REPLACE(
                     JSON_EXTRACT_SCALAR(payload, '$.pull_request.body'),
                     r'\\r\\n|\\r|\\n',
                     ' '),
                 r'\s{2,}',
                 ' ')) as body
    
    FROM `githubarchive.day.{year_prefix_wildcard}`
        WHERE _TABLE_SUFFIX IN {year_suffix_month_day}
        AND repo.name in {repo_names}
        AND type = 'PullRequestEvent'
    """

    PR_QUERY = bq_helper.bq_add_query_params(PR_QUERY, QUERY_PARAMS)
    qsize = GH_BQ_CLIENT.estimate_query_size(PR_QUERY)
    _logger.info("Retrieving GH Pull Requests. Query cost in GB={qc}".format(qc=qsize))

    prs_df = GH_BQ_CLIENT.query_to_pandas(PR_QUERY)
    if prs_df.empty:
        _logger.warn("No pull requests present for given time duration.")
    else:
        _logger.info("Total pull requests retrieved: {n}".format(n=len(prs_df)))

        prs_df.created_at = pd.to_datetime(prs_df.created_at)
        prs_df.updated_at = pd.to_datetime(prs_df.updated_at)
        prs_df.closed_at = pd.to_datetime(prs_df.closed_at)
        prs_df = prs_df.loc[prs_df.groupby("url").updated_at.idxmax(skipna=False)].reset_index(
            drop=True
        )
        _logger.info("Total pull requests after deduplication: {n}".format(n=len(prs_df)))

    _logger.info("\n")

    _logger.info("Merging issues and pull requests datasets")
    cols = issues_df.columns
    df = pd.concat([issues_df, prs_df], axis=0, sort=False, ignore_index=True).reset_index(
        drop=True
    )
    df = df[cols]

    if df.empty:
        _logger.warn("Nothing to predict today :)")
        sys.exit(0)

    _logger.info("Creating description column for NLP")
    df["description"] = df["title"].fillna(value="").map(str) + " " + df["body"].fillna(value="")
    columns = ["title", "body"]
    df.drop(columns, inplace=True, axis=1)
    return df


def run_inference(df, CVE_MODEL_TYPE="bert") -> pd.DataFrame:
    if "torch" not in CVE_MODEL_TYPE:
        from models.run_tf_models import run_bert_tensorflow_model, run_gru_cve_model

        if CVE_MODEL_TYPE == "gru":
            df = run_gru_cve_model(df)

        if CVE_MODEL_TYPE == "bert":
            df = run_bert_tensorflow_model(df)
    else:
        from models.run_torch_model import run_torch_cve_model_bert

        df = run_torch_cve_model_bert(df)
    return df


def upload_inference_results_s3(triage_results_dir):
    """Upload the generated inference results .csv to S3."""
    # ======= UPLOADING INFERENCE DATASETS TO S3 BUCKET ========
    _logger.info("----- UPLOADING INFERENCE DATASETS TO S3 BUCKET  -----")
    s3_obj = aws.S3_OBJ
    bucket_name = cc.S3_BUCKET_NAME_INFERENCE
    s3_bucket = s3_obj.Bucket(bucket_name)

    _logger.info("Uploading Saved Model Assets to S3 Bucket")
    aws.s3_upload_folder(
        folder_path=triage_results_dir,
        s3_bucket_obj=s3_bucket,
        prefix="triaged_datasets_openshift",
    )


if __name__ == "__main__":
    main()
    _logger.info("All done!")
