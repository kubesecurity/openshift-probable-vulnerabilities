from utils import aws_utils as aws
from utils import cloud_constants as cc
import os
import daiquiri
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-ts", "--triage-subfolder", type=str,
                    help="The specific triage sub-folder to upload to S3. Folder format is YYYYMMDD-YYYYMMDD.")
args = parser.parse_args()
TRIAGE_SUBFOLDER = args.triage_subfolder

daiquiri.setup(level=logging.INFO)
_logger = daiquiri.getLogger(__name__)

# No need to change this
s3_obj = aws.S3_OBJ
bucket_name = cc.S3_BUCKET_NAME
s3_bucket = s3_obj.Bucket(bucket_name)

# No need to change this
BASE_TRIAGE_DIR = './triaged_datasets'

# CHANGE NEEDED
# You need to specify the folder you want to upload back to S3 after manual triage
# This should be a sub-folder under triaged-datasets in your directory
# either modify it here or use the -ts or --triage-subfolder argument when running the script.
NEW_TRIAGE_SUBDIR = TRIAGE_SUBFOLDER or '20190526-20190528'

# No need to change this
TRIAGE_PATH = os.path.join(BASE_TRIAGE_DIR, NEW_TRIAGE_SUBDIR)


if __name__ == '__main__':
    _logger.info('Uploading Saved Model Assets to S3 Bucket')
    aws.s3_upload_folder(folder_path=TRIAGE_PATH,
                         s3_bucket_obj=s3_bucket, prefix='triaged_datasets')