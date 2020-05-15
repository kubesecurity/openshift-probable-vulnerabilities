### How To Run Tensorflow 1.14 training job using bert-tensorflow implementation on EMR

* Have a look at the "emr_bootstrap_train.sh" file that is present in this folder and modify it
to add any packages that you need installed for your changes work
* Note that as of this writing (EMR 5.29) the latest version of Tensorflow available on EMR is 1.14,
we run 1.15.x everywhere but not in this bootstrap as it's one of the EMR applications installed in the payload.
* Make sure all your data is present in S3.
* Zip up the source code and upload that zip to S3 as well.
* Make sure to import your aws keys under the environment variables `AWS_S3_ACCESS_KEY_ID` and `AWS_S3_SECRET_ACCESS_KEY`.
Then run the job submission script with `python submit_emr_job_train.py`, with changes to the data locations:

```python
        s3_bucket = os.environ.get('AWS_S3_BUCKET_NAME', 'avgupta-dev-emr-jobs') # Change these
        s3_key = os.environ.get('TRAINING_CODE_KEY', 'pytorch_train_job.zip')
        s3_uri = 's3://{bucket}/{key}'.format(bucket=s3_bucket, key=s3_key)
        s3_bootstrap_uri = 's3://{bucket}/emr_bootstrap_pytorch.sh'.format(bucket=s3_bucket)
        ...
        {
            'Name': 'setup - copy data',
            'ActionOnFailure': 'TERMINATE_CLUSTER',
            'HadoopJarStep': {
                'Jar': 'command-runner.jar',
                'Args': ['aws', 's3', 'cp', 's3://avgupta-dev-emr-jobs/train.tsv', '/home/hadoop/'] # Change this
            }
        },
        {
            'Name': 'setup - copy data',
            'ActionOnFailure': 'TERMINATE_CLUSTER',
            'HadoopJarStep': {
                'Jar': 'command-runner.jar',
                'Args': ['aws', 's3', 'cp', 's3://avgupta-dev-emr-jobs/dev.tsv', '/home/hadoop/'] # Change this.
            }
        }
```

## Local deployment

If you want to train locally (not recommended), you need to setup a bunch of environment variables
with the appropriate data downloaded, have a look at model_inference_triage_pipeline/models/utils/cloud_constants.py for this.
The variables at the time of writing are:
```python
GOOGLE_APPLICATION_CREDENTIALS
BIGQUERY_CREDENTIALS_FILEPATH
GOKUBE_REPO_LIST
KNATIVE_REPO_LIST
KUBEVIRT_REPO_LIST
BASE_BERT_UNCASED_PATH
P2BERT_CVE_MODEL_WEIGHTS_PATH
P1GRU_SEC_MODEL_TOKENIZER_PATH
P1GRU_SEC_MODEL_WEIGHTS_PATH
P1GRU_CVE_MODEL_TOKENIZER_PATH
P1GRU_CVE_MODEL_WEIGHTS_PATH
```