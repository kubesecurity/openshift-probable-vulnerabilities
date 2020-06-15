import json
import logging
import os
from datetime import datetime

import daiquiri
import pandas as pd
import requests

from utils import cloud_constants as cc

daiquiri.setup(level=logging.INFO)
log = daiquiri.getLogger(__name__)

INSERT_API_PATH = "/api/v1/pcve"
API_FULL_PATH = "{host}{api}".format(host=cc.OSA_API_SERVER_URL, api=INSERT_API_PATH)

S3_RAW_PATH = "s3://{bucket_name}/triaged_datasets/{triage_dir}/{failed_dataset_filename}"
FILE_NAME ="failed_to_insert_{subdir}_{ecosystem}.csv"

_failing_list = []


def _report_failures(df: pd.DataFrame, start_time: datetime, end_time: datetime, s3_upload: bool, ecosystem: str):
    """Save failed record."""
    if len(_failing_list) == 0:
        log.info("Successfully ingested all records")
    else:
        failed_list_str = ",".join(_failing_list)
        log.error("Failed to insert {count} data : {data}".format(count=len(_failing_list),data=failed_list_str))

        subdir = "-".join([start_time.format("YYYYMMDD"), end_time.format("YYYYMMDD")])
        results_dir = os.path.join(cc.BASE_TRIAGE_DIR, subdir)

        failed_dataset_filename = FILE_NAME.format(subdir=subdir, ecosystem=ecosystem)

        failed_data = df[df['url'].isin(_failing_list)]
        logging.info("failed data count :{count}".format(count=str(failed_data.shape[0])))

        if not s3_upload:
            failed_data_file_path = os.path.join(results_dir, failed_dataset_filename)
            log.info("Saving data those failed to insert at {}".format(failed_data_file_path))
            failed_data.to_csv(failed_data_file_path, index=False)
        else:
            s3_path = S3_RAW_PATH.format(bucket_name=cc.S3_BUCKET_NAME_INFERENCE,  triage_dir=subdir,
                                         failed_dataset_filename=failed_dataset_filename)
            log.info("Saving data those failed to insert to S3: {}".format(s3_path))
            failed_data.to_csv(s3_path, index=False)

        log.info("Failed data saved successfully.")


def _insert_df(df: pd.DataFrame, url:str):
    """Call API server and insert data."""
    logging.info("inside _insert_df")
    objs = df.to_dict(orient='records')
    for obj in objs:
        result = requests.post(url, json=obj)
        log.debug('Got response {} for {}'.format(result.status_code, obj))
        if result.status_code != 200:
            log.error('Error response: {}, msg: {}, data: {}'.format(result.status_code,result.json()["message"], json.dumps(obj)))
            _failing_list.append(obj["url"])


def _get_status_type(status: str) -> str:
    if status.lower() in ['opened', 'closed', 'reopened']:
        return status.upper()
    else:
        return "OTHER"


def _update_df(df: pd.DataFrame, ecosystem: str) -> pd.DataFrame:
    df = df[df.cve_model_flag == 1]
    df['status'] = df.apply(lambda x: _get_status_type(x['status']), axis=1)
    df['ecosystem'] = ecosystem.upper()
    df['probable_cve'] = df.apply(lambda x: _get_probabled_cve(x['cve_model_flag']), axis=1)
    return df.where(pd.notnull(df), None)


def _get_probabled_cve(cve_model_flag: int) -> bool:
    return True if cve_model_flag is not None and cve_model_flag == 1 else False


def save_data_to_db(df: pd.DataFrame, start_time: datetime, end_time: datetime, s3_upload, ecosystem):
    """Save data to database using api server."""
    logging.info("totoal data count :{count}".format(count=str(df.shape[0])))

    updated_df = _update_df(df, ecosystem)
    logging.info("pccve data count :{count}".format(count=str(updated_df.shape[0])))

    logging.info("update df completed")

    _insert_df(updated_df, API_FULL_PATH)

    # Save data to csv file those are failed to ingest
    _report_failures(df, start_time, end_time, ecosystem, s3_upload)


# if __name__ == '__main__':
#
#     logging.info("reading data")
#     df = pd.read_csv("bert_model_inference_probable_cves_20200525-20200531_openshift.csv")
#
#     logging.info("data count :{count}".format(count=str(df.shape[0])))
#     # print(df.head(10).to_string())
#
#     present_time = arrow.now()
#     day_count = 3  # Gets 3 days of previous data including YESTERDAY
#     start_time = present_time.shift(days=-day_count)
#     end_time = present_time.shift(days=-1)
#     s3_upload = True
#     ecosystem = "openshift"
#     save_data_to_db(df, start_time, end_time, ecosystem, s3_upload)

