"""Contains the osa api and other contants."""

import os


BASE_TRIAGE_DIR = os.environ.get("BASE_TRIAGE_DIR", "/data_assets/triaged_datasets")
NEW_TRIAGE_SUBDIR = "{stat_time}-{end_time}"
FAILED_TO_INSERT = "failed_to_insert"
OUTPUT_FILE_NAME = "{file_prefix}_{data_type}_{triage_dir}_{ecosystem}.csv"
FULL_OUTPUT = "full_output"
PROBABLE_SECURITY_AND_CVES = "probable_security_and_cves"
PROBABLE_CVES = "probable_cves"
MAX_STRING_LEN_FOR_CSV_EXPORT = int(os.environ.get("MAX_STRING_LEN_FOR_CSV_EXPORT", 2000))

# API server Constants
OSA_API_SERVER_HOST = os.environ.get("OSA_API_SERVER_HOST", "osa-api-server")
OSA_API_SERVER_PORT = os.environ.get("OSA_API_SERVER_PORT", 5000)
OSA_API_SERVER_URL = 'http://{host}:{port}'.format(host=OSA_API_SERVER_HOST, port=OSA_API_SERVER_PORT)
DATA_INSERT_CONCURRENCY = int(os.environ.get("DATA_INSERT_CONCURRENCY", 10))
SKIP_INSERT_API_CALL = os.environ.get("SKIP_INSERT_API_CALL", "false").lower() == "true"
