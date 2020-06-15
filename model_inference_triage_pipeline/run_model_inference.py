import argparse
import logging
import sys
import textwrap
import warnings

import arrow
import daiquiri
import pandas as pd

from utils import aws_utils as aws
from utils import cloud_constants as cc
from utils.bq_utils import get_bq_data_for_inference
from utils.storage_utils import write_output_csv
from utils.api_util import save_data_to_db

daiquiri.setup(level=logging.INFO)
_logger = daiquiri.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)


def main():
    """The main program logic."""
    parser = get_argument_parser()
    args = parser.parse_args()

    ECOSYSTEM = args.eco_system.lower()
    COMPUTE_DEVICE = args.compute_device.lower()

    if COMPUTE_DEVICE != "cpu":
        import torch

        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    SEC_MODEL_TYPE = args.security_filter_model.lower()
    if SEC_MODEL_TYPE != "" and SEC_MODEL_TYPE != "gru":
        _logger.error("GRU is the only supported security model type.")
        sys.exit(0)
    CVE_MODEL_TYPE = args.probable_cve_model.lower()
    S3_UPLOAD = args.s3_upload_cves

    day_count, start_time, end_time, date_range = setup_dates_for_triage(
        args.days_since_yday, args.start_date, args.end_date
    )

    df = get_bq_data_for_inference(ECOSYSTEM, day_count, date_range)
    df = run_inference(df, CVE_MODEL_TYPE)
    write_output_csv(
        start_time,
        end_time,
        cve_model_type=CVE_MODEL_TYPE,
        ecosystem=ECOSYSTEM,
        df=df,
        s3_upload=S3_UPLOAD,
    )

    # Save data to database using api server
    save_data_to_db(df, start_time, end_time, S3_UPLOAD, ECOSYSTEM)


# noinspection PyTypeChecker
def get_argument_parser():
    """Defines all the command line arguments for the program."""
    description: str = textwrap.dedent(
        """
        This script can be used to run our AI models for probable vulnerability predictions.
        Check usage patterns below.
        """
    )
    epilog: str = textwrap.dedent(
        """
         Usage patterns for inference script
         -----------------------------------
         The -days flag should be used with number of prev days data you want to pull

         1. GRU Models (Baseline): python run_model_inference.py -days=7 -device=gpu -sec-model=gru -cve-model=gru
         2. BERT Model (CVE): python run_model_inference.py -days=7 -device=gpu -sec-model=gru -cve-model=bert
         3. CPU inference:
                python run_model_inference.py -days=7 -device=cpu -sec-model=gru -cve-model=gru
         """
    )
    parser = argparse.ArgumentParser(
        prog="python",
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        choices=["gru", "GRU", "bert", "BERT", "bert_torch", "BERT_TORCH"],
        help="The AI Model to use for probable CVE predictions - Model 2",
    )

    parser.add_argument(
        "-s3-upload",
        "--s3-upload-cves",
        type=bool,
        default=False,
        choices=[False, True],
        help=(
            "Uploads inference CSVs to S3 bucket - should have write access to the appropriate S3 bucket."
        ),
    )

    parser.add_argument(
        "-sd",
        "--start-date",
        default="",
        help="If running for a custom interval, set this and the [end-date] in yyyy-mm-dd format.",
    )

    parser.add_argument(
        "-ed",
        "--end-date",
        default="",
        help="If running for a custom interval, set this and the [start-date] in yyyy-mm-dd format.",
    )

    return parser


def setup_dates_for_triage(days_since_yday, start_date_user, end_date_user):
    """Sets up the date range for data retireval."""
    # ======= DATES SETUP FOR GETTING GITHUB BQ DATA ========
    _logger.info("----- DATES SETUP FOR GETTING GITHUB BQ DATA -----")
    if start_date_user != "" and end_date_user != "":
        start_time = arrow.get(start_date_user, "YYYY-MM-DD")
        end_time = arrow.get(end_date_user, "YYYY-MM-DD")
        day_count = (end_time - start_time).days
        date_range = [
            dt.format("YYYYMMDD") for dt in arrow.Arrow.range("day", start_time, end_time)
        ]
        return day_count, start_time, end_time, date_range

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


def run_inference(df, CVE_MODEL_TYPE="bert") -> pd.DataFrame:
    if cc.S3_MODEL_REFRESH.lower() == "true":
        aws.s3_download_folder(aws.S3_OBJ.Bucket(cc.S3_BUCKET_NAME_MODEL), "model_assets", "/")

    if "torch" not in CVE_MODEL_TYPE.lower():
        from models.run_tf_models import (
            run_bert_tensorflow_model,
            run_gru_cve_model,
            run_tensorflow_security_classifier,
        )

        df = run_tensorflow_security_classifier(df)
        if CVE_MODEL_TYPE == "gru":
            df = run_gru_cve_model(df)

        if CVE_MODEL_TYPE == "bert":
            df = run_bert_tensorflow_model(df)
    else:
        from models.run_torch_model import run_torch_cve_model_bert
        from models.run_tf_models import run_tensorflow_security_classifier

        # Re-use the GRU based security/non-security classifier then pipe its output to the new BERT model.
        df = run_torch_cve_model_bert(run_tensorflow_security_classifier(df))
    return df


if __name__ == "__main__":
    main()
    _logger.info("All done!")
