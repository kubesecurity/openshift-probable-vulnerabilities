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
    _logger.info('Downloading Saved Model Assets from S3 Bucket')
    aws.s3_download_folder(s3_bucket_obj=s3_bucket,
                           bucket_dir_prefix='model_assets', download_path='./models')
