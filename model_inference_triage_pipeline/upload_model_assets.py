"""Module to handle model file upload to Object store."""

from utils import aws_utils as aws
from utils import cloud_constants as cc
import daiquiri
import logging

daiquiri.setup(level=logging.INFO)
_logger = daiquiri.getLogger(__name__)

s3_obj = aws.S3_OBJ
bucket_name = cc.S3_BUCKET_NAME
s3_bucket = s3_obj.Bucket(bucket_name)


if __name__ == '__main__':
    _logger.info('Uploading Saved Model Assets to S3 Bucket')
    aws.s3_upload_folder(folder_path='./models/model_assets/gokube-phase1-jun19',
                         s3_bucket_obj=s3_bucket, prefix='model_assets')
    aws.s3_upload_folder(folder_path='./models/model_assets/gokube-phase2',
                         s3_bucket_obj=s3_bucket, prefix='model_assets')
