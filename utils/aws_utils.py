from utils import cloud_constants as cc
import boto3
import botocore
import os
from pathlib import Path
import daiquiri
import logging

daiquiri.setup(level=logging.INFO)
_logger = daiquiri.getLogger(__name__)

_aws_key_id = cc.AWS_S3_ACCESS_KEY_ID
_aws_secret_key = cc.AWS_S3_SECRET_ACCESS_KEY
_aws_region = cc.AWS_S3_REGION

session = boto3.session.Session(aws_access_key_id=_aws_key_id,
                                aws_secret_access_key=_aws_secret_key,
                                region_name=_aws_region)

S3_OBJ = session.resource('s3', config=botocore.client.Config(signature_version='s3v4'),
                          use_ssl=True)


def s3_download_folder(s3_bucket_obj, bucket_dir_prefix='', download_path='./', ):
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for obj in s3_bucket_obj.objects.filter(Prefix=bucket_dir_prefix):
        s3_object = obj.key
        _logger.info('Downloading object: {obj}'.format(obj=s3_object))
        if not s3_object.endswith("/"):
            path, file = os.path.split(s3_object)
            if not os.path.exists(os.path.join(download_path, path)):
                os.makedirs(os.path.join(download_path, path))
            s3_bucket_obj.download_file(
                s3_object, os.path.join(download_path, s3_object))
        else:
            if not os.path.exists(os.path.join(download_path, s3_object)):
                os.makedirs(os.path.join(download_path, s3_object))


def s3_upload_folder(folder_path, s3_bucket_obj, prefix=''):
    resolved_path = Path(folder_path).resolve()
    parent_dir = resolved_path.parent
    for root, _, filenames in os.walk(resolved_path):
        for filename in filenames:
            if root != '.':
                s3_dest = os.path.join(prefix,
                                       Path(root).relative_to(parent_dir), filename)
            else:
                s3_dest = os.path.join(prefix, filename)
            _logger.info('Uploading to: {d}'.format(d=s3_dest))
            s3_bucket_obj.upload_file(os.path.join(root, filename), s3_dest)
