#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script spawns a spark emr cluster on AWS and submits a job to run the given src code.

Copyright © 2018 Red Hat Inc.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import logging
import boto3
from time import gmtime, strftime
import daiquiri
import os

daiquiri.setup(level=logging.DEBUG)
_logger = daiquiri.getLogger(__name__)

COMPONENT_PREFIX = "go-cve"
AWS_S3_ACCESS_KEY_ID = os.environ.get('AWS_S3_ACCESS_KEY_ID', '')
AWS_S3_SECRET_ACCESS_KEY = os.environ.get('AWS_S3_SECRET_ACCESS_KEY', '')
_logger.info(AWS_S3_ACCESS_KEY_ID)
_logger.info(AWS_S3_SECRET_ACCESS_KEY)

def submit_job():
    """Submit new job with specified parameters."""
    str_cur_time = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    # S3 bucket/key, where the input spark job ( src code ) will be uploaded
    s3_bucket = os.environ.get('AWS_S3_BUCKET_NAME', 'avgupta-dev-emr-jobs')
    s3_key = os.environ.get('TRAINING_CODE_KEY', 'tf_train_job.zip')
    s3_uri = 's3://{bucket}/{key}'.format(bucket=s3_bucket, key=s3_key)
    s3_bootstrap_uri = 's3://{bucket}/emr_bootstrap_train.sh'.format(bucket=s3_bucket)

    # S3 bucket/key, where the spark job logs will be maintained
    s3_log_bucket = 'avgupta-dev-emr-jobs'
    s3_log_key = '{}_spark_emr_log_'.format(str_cur_time)
    s3_log_uri = 's3://{bucket}/{key}'.format(bucket=s3_log_bucket, key=s3_log_key)

    _logger.debug("bootstrap action AWS S3 URI {} ...".format(s3_bootstrap_uri))
    _logger.debug("Source code AWS S3 URI {} ...".format(s3_uri))
    _logger.debug("Starting spark emr cluster and submitting the jobs ...")
    emr_client = boto3.client('emr',
                              aws_access_key_id=AWS_S3_ACCESS_KEY_ID,
                              aws_secret_access_key=AWS_S3_SECRET_ACCESS_KEY,
                              region_name='us-east-1')

    response = emr_client.run_job_flow(
        Name=COMPONENT_PREFIX + "_" + str_cur_time,
        LogUri=s3_log_uri,
        ReleaseLabel='emr-5.29.0',
        Instances={
            'KeepJobFlowAliveWhenNoSteps': False,
            'TerminationProtected': False,
            'InstanceGroups': [
                {
                    'Name': '{}_master_group'.format(COMPONENT_PREFIX),
                    'InstanceRole': 'MASTER',
                    'InstanceType': 'p3.2xlarge',
                    'InstanceCount': 1,
                    'Configurations': [
                        {
                            "Classification": "hadoop-env",
                            "Properties": {},
                            "Configurations": [
                                {
                                    "Classification": "export",
                                    "Configurations": [],
                                    "Properties": {
                                        "PYTHONPATH": "/home/hadoop/",
                                        "LC_ALL": "en_US.UTF-8",
                                        "LANG": "en_US.UTF-8",
                                        "AWS_S3_ACCESS_KEY_ID": AWS_S3_ACCESS_KEY_ID,
                                        "AWS_S3_SECRET_ACCESS_KEY": AWS_S3_SECRET_ACCESS_KEY,
                                        "AWS_S3_BUCKET_NAME": s3_bucket,
                                    }
                                }
                            ]
                        }
                    ]
                }
            ],
        },
        BootstrapActions=[
            {
                'Name': 'Metadata setup',
                'ScriptBootstrapAction': {
                    'Path': s3_bootstrap_uri
                }
            }
        ],
        Steps=[
            {
                'Name': 'Setup Debugging',
                'ActionOnFailure': 'TERMINATE_CLUSTER',
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': ['state-pusher-script']
                }
            },
            {
                'Name': 'setup - copy files',
                'ActionOnFailure': 'TERMINATE_CLUSTER',
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': ['aws', 's3', 'cp', s3_uri, '/home/hadoop/']
                }
            },
            {
                'Name': 'setup - copy data',
                'ActionOnFailure': 'TERMINATE_CLUSTER',
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': ['aws', 's3', 'cp', 's3://avgupta-dev-emr-jobs/GH_complete_labeled_issues_prs - preprocessed.csv', '/home/hadoop/']
                }
            },
            {
                 'Name': 'setup - unzip files',
                 'ActionOnFailure': 'TERMINATE_CLUSTER',
                 'HadoopJarStep': {
                     'Jar': 'command-runner.jar',
                     'Args': ['unzip', '/home/hadoop/' + s3_key, '-d', '/home/hadoop']
                 }
            },
            {
                'Name': 'Run training job',
                'ActionOnFailure': 'TERMINATE_CLUSTER',
                'HadoopJarStep': {
                    'Jar': 'command-runner.jar',
                    'Args': ["/home/hadoop/.local/bin/ipython",
                             '/home/hadoop/bert_cve_train.py']
                }
            }
        ],
        Applications=[{'Name': 'TensorFlow'}],
        VisibleToAllUsers=True,
        JobFlowRole='EMR_EC2_DefaultRole',
        ServiceRole='EMR_DefaultRole'
    )

    output = {}
    if response.get('ResponseMetadata', {}).get('HTTPStatusCode') == 200:

        output['training_job_id'] = response.get('JobFlowId')
        output['status'] = 'work_in_progress'
        output[
            'status_description'] = "The training is in progress. Please check the given " \
                                    "training job after some time."
    else:
        output['training_job_id'] = "Error"
        output['status'] = 'Error'
        output['status_description'] = "Error! The job/cluster could not be created!"
        _logger.debug(response)

    return output


if __name__ == "__main__":
    print(submit_job())
