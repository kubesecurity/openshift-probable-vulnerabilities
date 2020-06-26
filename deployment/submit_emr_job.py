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
import argparse

daiquiri.setup(level=logging.DEBUG)
_logger = daiquiri.getLogger(__name__)

# Do not pick these from constants file, this script is a job scheduler and should be able to run
# independent of source of module.
COMPONENT_PREFIX = os.environ.get("COMPONENT_PREFIX", "go-cve")
S3_MODEL_ACCESS_KEY_ID = os.environ.get("S3_MODEL_ACCESS_KEY_ID", "")
S3_MODEL_SECRET_ACCESS_KEY = os.environ.get("S3_MODEL_SECRET_ACCESS_KEY", "")


def submit_job(steps, s3_source_uri, s3_bootstrap_uri, source_bucket, s3_log_bucket):
    """Submit new job with specified parameters."""
    str_cur_time = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    s3_log_key = "{}_spark_emr_log_".format(str_cur_time)
    s3_log_uri = "s3://{bucket}/{key}".format(bucket=s3_log_bucket, key=s3_log_key)

    # S3 bucket/key, where the input spark job ( src code ) will be uploaded
    _logger.debug("bootstrap action AWS S3 URI {} ...".format(s3_bootstrap_uri))
    _logger.debug("Source code AWS S3 URI {} ...".format(s3_source_uri))
    _logger.debug("Starting spark emr cluster and submitting the jobs ...")
    emr_client = boto3.client(
        "emr",
        aws_access_key_id=S3_MODEL_ACCESS_KEY_ID,
        aws_secret_access_key=S3_MODEL_SECRET_ACCESS_KEY,
        region_name="us-east-1",
    )

    response = emr_client.run_job_flow(
        Name=COMPONENT_PREFIX + "_" + str_cur_time,
        LogUri=s3_log_uri,
        ReleaseLabel="emr-5.29.0",
        Instances={
            "KeepJobFlowAliveWhenNoSteps": False,
            "TerminationProtected": False,
            "InstanceGroups": [
                {
                    "Name": "{}_master_group".format(COMPONENT_PREFIX),
                    "InstanceRole": "MASTER",
                    "InstanceType": "p3.2xlarge",
                    "InstanceCount": 1,
                    "Configurations": [
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
                                        "S3_MODEL_ACCESS_KEY_ID": S3_MODEL_ACCESS_KEY_ID,
                                        "S3_MODEL_SECRET_ACCESS_KEY": S3_MODEL_SECRET_ACCESS_KEY,
                                        "AWS_S3_BUCKET_NAME": source_bucket,
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        BootstrapActions=[
            {"Name": "Metadata setup", "ScriptBootstrapAction": {"Path": s3_bootstrap_uri}}
        ],
        Steps=steps,
        Applications=[{"Name": "TensorFlow"}],
        VisibleToAllUsers=True,
        JobFlowRole="EMR_EC2_DefaultRole",
        ServiceRole="EMR_DefaultRole",
    )

    output = {}
    if response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 200:

        output["training_job_id"] = response.get("JobFlowId")
        output["status"] = "work_in_progress"
        output["status_description"] = (
            "The training is in progress. Please check the given " "training job after some time."
        )
    else:
        output["training_job_id"] = "Error"
        output["status"] = "Error"
        output["status_description"] = "Error! The job/cluster could not be created!"
        _logger.debug(response)

    return output


def get_tensorflow_job_steps(s3_uri, s3_source_code_key):
    return [
        {
            "Name": "Setup Debugging",
            "ActionOnFailure": "TERMINATE_CLUSTER",
            "HadoopJarStep": {"Jar": "command-runner.jar", "Args": ["state-pusher-script"]},
        },
        {
            "Name": "setup - copy files",
            "ActionOnFailure": "TERMINATE_CLUSTER",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": ["aws", "s3", "cp", s3_uri, "/home/hadoop/"],
            },
        },
        {
            "Name": "setup - copy data",
            "ActionOnFailure": "TERMINATE_CLUSTER",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": [
                    "aws",
                    "s3",
                    "cp",
                    "s3://avgupta-dev-emr-jobs/GH_complete_labeled_issues_prs - preprocessed.csv",
                    "/home/hadoop/",
                ],
            },
        },
        {
            "Name": "setup - unzip files",
            "ActionOnFailure": "TERMINATE_CLUSTER",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": ["unzip", "/home/hadoop/" + s3_source_code_key, "-d", "/home/hadoop"],
            },
        },
        {
            "Name": "Run training job",
            "ActionOnFailure": "TERMINATE_CLUSTER",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": ["/home/hadoop/.local/bin/ipython", "/home/hadoop/bert_cve_train.py"],
            },
        },
    ]


def get_torch_job_steps(s3_source_uri, s3_source_code_key):
    return [
        {
            "Name": "Setup Debugging",
            "ActionOnFailure": "TERMINATE_CLUSTER",
            "HadoopJarStep": {"Jar": "command-runner.jar", "Args": ["state-pusher-script"]},
        },
        {
            "Name": "setup - copy files",
            "ActionOnFailure": "TERMINATE_CLUSTER",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": ["aws", "s3", "cp", s3_source_uri, "/home/hadoop/"],
            },
        },
        {
            "Name": "setup - copy data (train)",
            "ActionOnFailure": "TERMINATE_CLUSTER",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": [
                    "aws",
                    "s3",
                    "cp",
                    "s3://avgupta-dev-emr-jobs/train.tsv",
                    "/home/hadoop/",
                ],
            },
        },
        {
            "Name": "setup - copy data (dev)",
            "ActionOnFailure": "TERMINATE_CLUSTER",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": ["aws", "s3", "cp", "s3://avgupta-dev-emr-jobs/dev.tsv", "/home/hadoop/"],
            },
        },
        {
            "Name": "setup - unzip files",
            "ActionOnFailure": "TERMINATE_CLUSTER",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": ["unzip", "/home/hadoop/" + s3_source_code_key, "-d", "/home/hadoop"],
            },
        },
        {
            "Name": "Run training job",
            "ActionOnFailure": "TERMINATE_CLUSTER",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": [
                    "python3",
                    "/home/hadoop/bert_pytorch.py",
                    "--data_dir=/home/hadoop",
                    "--model_name_or_path=bert-base-uncased",
                    "--task_name=sst-2",
                    "--output_dir=/mnt2/models/model_assets/gokube-phase2/saved_models/pytorch-cve-warmup",
                    "--max_seq_length=512",
                    "--cache_dir=/mnt1/cache/",
                    "--do_lower_case",
                    "--do_train",
                    "--do_eval",
                    "--evaluate_during_training",
                    "--save_steps=2000",
                    "--learning_rate=5e-5",
                    "--weight_decay=0.01",
                    "--per_gpu_train_batch_size=12",
                    "--num_train_epochs=10.0",
                    "--eval_all_checkpoints",
                    "--overwrite_output_dir",
                ],
            },
        },
        {
            "Name": "Upload model to S3",
            "ActionOnFailure": "TERMINATE_CLUSTER",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": [
                    "aws",
                    "s3",
                    "sync",
                    "/mnt2/models/model_assets/gokube-phase2/saved_models/pytorch-cve-warmup",
                    "s3://avgupta-dev-emr-jobs/pytorch-cve-warmup-{}/".format(
                        strftime("%Y_%m_%d_%H_%M_%S", gmtime())
                    ),
                ],
            },
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Run BERT model training on EMR.")
    parser.add_argument(
        "-b",
        "--bootstrap-file",
        type=str,
        required=True,
        help="The name of the bootstrap actions file.",
    )
    parser.add_argument(
        "-m",
        "--model-type",
        type=str,
        default="torch",
        choices=["torch", "tensorflow"],
        required=True,
        help="The AI Model to train for probable CVE prediction.",
    )
    parser.add_argument(
        "-sb",
        "--source-bucket",
        type=str,
        required=True,
        help="The bucket containing the source code zip and bootstrap actions.",
    )
    parser.add_argument(
        "-lb",
        "--logs-bucket",
        type=str,
        required=True,
        help="The bucket where the EMR job logs will be written to.",
        default="avgupta-dev-emr-jobs",
    )
    parser.add_argument(
        "-sc",
        "--source-code-key",
        type=str,
        help="The filename of the source code zip, present on S3",
    )

    args = parser.parse_args()
    s3_source_uri = "s3://{bucket}/{key}".format(bucket=args.source_bucket, key=args.souce_code_key)
    s3_bootstrap_uri = "s3://{bucket}/{bootstrap}".format(
        bucket=args.source_bucket, bootstrap=args.bootstrap_file
    )
    steps = None
    if args.model_type == "torch":
        steps = get_torch_job_steps(s3_source_uri, args.source_code_key)
    else:
        steps = get_tensorflow_job_steps(s3_source_uri, args.source_code_key)
    # This is output of the program and not a log message, so printing to ensure output goes to stdout.
    print(submit_job(steps, s3_source_uri, s3_bootstrap_uri, args.source_bucket, args.logs_bucket))


if __name__ == "__main__":
    main()
