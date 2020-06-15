"""Ingests historical CSV data into DB."""

import json
import os
from datetime import datetime

import daiquiri
import pandas as pd
import requests

from utils import cloud_constants as cc
from utils.storage_utils import get_file_prefix, save_data_to_csv

log = daiquiri.getLogger(__name__)  # pylint: disable=invalid-name

INSERT_API_PATH = "/api/v1/pcve"
API_FULL_PATH = "{host}{api}".format(host=cc.OSA_API_SERVER_URL, api=INSERT_API_PATH)

_failing_list: []


def _report_failures(df: pd.DataFrame, triage_subdir: str, s3_upload: bool, ecosystem: str):
    """Save failed record."""
    if len(_failing_list) == 0:
        log.info("Successfully ingested all records")
    else:
        failed_list_str = ",".join(_failing_list)
        log.error("Failed to insert {count} data : {data}".format(count=len(_failing_list), data=failed_list_str))

        failed_data = df[df['url'].isin(_failing_list)]
        save_data_to_csv(failed_data, s3_upload, cc.FAILED_TO_INSERT, triage_subdir, ecosystem, cc.PROBABLE_CVES)

        log.info("Failed data saved successfully.")


def _insert_df(df: pd.DataFrame, url: str):
    """Call API server and insert data."""
    log.info("inside _insert_df")
    objs = df.to_dict(orient='records')
    for obj in objs:
        result = requests.post(url, json=obj)
        log.debug('Got response {} for {}'.format(result.status_code, obj))
        if result.status_code != 200:
            log.error('Error response: {}, msg: {}, data: {}'
                      .format(result.status_code, result.json()["message"], json.dumps(obj)))
            _failing_list.append(obj["url"])


def _get_status_type(status: str) -> str:
    if status.lower() in ['opened', 'closed', 'reopened']:
        return status.upper()
    else:
        return "OTHER"


def _get_probabled_cve(cve_model_flag: int) -> bool:
    return True if cve_model_flag is not None and cve_model_flag == 1 else False


def _update_df(df: pd.DataFrame) -> pd.DataFrame:
    df['ecosystem'] = df['ecosystem'].str.upper()
    df['status'] = df.apply(lambda x: _get_status_type(x['status']), axis=1)

    if 'cve_model_flag' not in df:
        df['probable_cve'] = True
    else:
        df['probable_cve'] = df.apply(lambda x: _get_probabled_cve(x['cve_model_flag']), axis=1)

    return df.where(pd.notnull(df), None)


def save_data_to_db(start_time: datetime, end_time: datetime, cve_model_type, s3_upload, ecosystem):
    """Save probable cve data to db via api server"""
    triage_subdir = cc.NEW_TRIAGE_SUBDIR.format(stat_time=start_time.format("YYYYMMDD"),
                                                end_time=end_time.format("YYYYMMDD"))
    df = _read_probable_cve_data(triage_subdir, cve_model_type, s3_upload, ecosystem)
    if len(df) != 0:

        log.info("PCVE data count :{count}".format(count=str(df.shape[0])))
        updated_df = _update_df(df)
        log.info("update df completed")

        log.info("PCVE data count after update :{count}".format(count=str(df.shape[0])))

        _insert_df(updated_df, API_FULL_PATH)

        # Save data to csv file those are failed to ingest
        _report_failures(df, triage_subdir,  ecosystem, s3_upload)
    else:
        log.info("No PCVE records to save for {}".format(ecosystem))


def _read_probable_cve_data(triage_subdir, cve_model_type, s3_upload, ecosystem):

    triage_results_dir = os.path.join(cc.BASE_TRIAGE_DIR, triage_subdir)
    file_prefix = get_file_prefix(cve_model_type)
    filename = cc.OUTPUT_FILE_NAME.format(data_type=cc.PROBABLE_CVES, file_prefix=file_prefix,
                                          new_triage_subdir=triage_subdir, ecosystem=ecosystem)
    dataset = os.path.join(triage_results_dir, filename)

    if not s3_upload:
        log.info("Reading {} dataset from local folder:{}".format(cc.PROBABLE_CVES, dataset))
        df = pd.read_csv(dataset, index_col=None, header=0)
    else:
        s3_path = cc.S3_FILE_PATH.format(bucket_name=cc.S3_BUCKET_NAME_INFERENCE, triage_dir=triage_subdir,
                                         dataset_filename=filename)
        log.info("Reading {} dataset from s3:{}".format(cc.PROBABLE_CVES, s3_path))
        df = pd.read_csv(s3_path, index_col=None, header=0)
    return df
