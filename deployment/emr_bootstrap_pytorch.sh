#!/bin/bash
# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

set -ex

worker_path=/home/hadoop/worker_ip_file

echo "Finding ip addresses of all the nodes in the cluster.."

# remove file if already exists
rm -rf /tmp/worker_metadata

# gets ip addresses of the nodes in the cluster and saves in temp file
# uncomment these lines if doing distributed training.
#for LINE in `yarn node -list | grep RUNNING | cut -f1 -d:`
#do
  #nslookup $LINE | grep Add | grep -v '#' | cut -f 2 -d ' ' >> /tmp/worker_metadata
#done

# ip address of master node saved in temp file
echo $(hostname -i) >> /tmp/worker_metadata

# sorting node ip addresses
sort -n -t . -k 1,1 -k 2,2 -k 3,3 -k 4,4 /tmp/worker_metadata > $worker_path

rm -rf /tmp/worker_metadata

echo "Setting DEEPLEARNING_WORKERS_PATH, DEEPLEARNING_WORKERS_COUNT, DEEPLEARNING_WORKER_GPU_COUNT as environment variables"

sed -i '/export DEEPLEARNING_WORKERS_PATH=/d' ~/.bashrc
echo "export DEEPLEARNING_WORKERS_PATH="$worker_path >> ~/.bashrc

# keeping worker_count=number of nodes in the cluster
worker_count="$(wc -l < $worker_path)"

sed -i '/export DEEPLEARNING_WORKERS_COUNT=/d' ~/.bashrc
echo "export DEEPLEARNING_WORKERS_COUNT="$worker_count >> ~/.bashrc

# setting up number of gpus as env var (Assuming uniform cluster)
gpu_count="$(nvidia-smi -L | grep ^GPU | wc -l)"

sed -i '/export DEEPLEARNING_WORKER_GPU_COUNT=/d' ~/.bashrc
echo "export DEEPLEARNING_WORKER_GPU_COUNT="$gpu_count >> ~/.bashrc

source ~/.bashrc

if [ -z ${DEEPLEARNING_WORKERS_COUNT+x} ];
    then echo "DEEPLEARNING_WORKERS_COUNT is unset";
    else echo "DEEPLEARNING_WORKERS_COUNT is set to '$DEEPLEARNING_WORKERS_COUNT'";
fi

echo "DEEPLEARNING_WORKERS_PATH set to: " $DEEPLEARNING_WORKERS_PATH
echo "Number of GPUs: " $DEEPLEARNING_WORKER_GPU_COUNT

echo "Environment variables are set!"

export PATH="/usr/local/bin:$PATH"
python3 -m pip install --upgrade pip setuptools --user
python3 -m pip install text-normalizer==0.1.3 scikit-learn==0.22.1 seaborn --user
python3 -m pip install ipython ipykernel jupyter ipywidgets --upgrade --user
python3 -m pip install torch==1.2.0 bert-pytorch transformers tqdm tensorboardX --user
mkdir -p /home/hadoop/models/model_assets/gokube-phase2/saved_models

# Now set the PYTHONPATH
export PYTHONPATH='/home/hadoop'
