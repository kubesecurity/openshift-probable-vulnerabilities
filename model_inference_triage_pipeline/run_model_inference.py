import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
from utils import cloud_constants as cc
from utils import bq_client_helper as bq_helper
from utils import text_normalizer as tn
from utils import aws_utils as aws
from utils import bert_text_processor as btp
from models import bert_cve_classifier as bcvec
from models import security_dl_classifier as sdc
from models import cve_dl_classifier as cdc

import pandas as pd
import numpy as np
import arrow
import daiquiri
import logging
import gc
import os
import argparse
import textwrap
import tensorflow as tf

# Initial setup no need to change anything
parser = argparse.ArgumentParser(prog='python',
                                 description=textwrap.dedent('''\
                    This script can be used to run our
                    AI models for probable vulnerability predictions.
                    Check usage patterns below'''),
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=textwrap.dedent('''\
         Usage patterns for inference script
         -----------------------------------
            The -days flag should be used with number of prev days data you want to pull

            1. GRU Models (Baseline): python run_model_inference.py -days=7 -device=gpu -sec-model=gru -cve-model=gru 
            2. BERT Model (CVE): python run_model_inference.py -days=7 -device=gpu -sec-model=gru -cve-model=bert 
            3. CPU inference (not recommended for BERT): python run_model_inference.py -days=7 -device=cpu -sec-model=gru -cve-model=gru  
         '''))

parser.add_argument("-days", "--days-since-yday", type=int, default=7,
                    help="The number of days worth of data to retrieve from GitHub including yesterday")

parser.add_argument("-eco", "--eco-system", type=str, default="openshift",
                    choices=["openshift", "knative", "kubevirt"],
                    help="The eco-system to monitor")

parser.add_argument("-device", "--compute-device", type=str, default="gpu",
                    choices=["gpu", "cpu", "GPU", "CPU"],
                    help="[Not implemented should work automatically] The backend device to run the AI models on - GPU or CPU")

parser.add_argument("-sec-model", "--security-filter-model", type=str, default="gru",
                    choices=["gru", "GRU"],
                    help="The AI Model to use for security filtering - Model 1")

parser.add_argument("-cve-model", "--probable-cve-model", type=str, default="gru",
                    choices=["gru", "GRU", "bert", "BERT"],
                    help="The AI Model to use for probable CVE predictions - Model 2")   

parser.add_argument("-s3-upload", "--s3-upload-cves", type=bool, default=False,
                    choices=[False, True],
                    help="Uploads inference CSVs to S3 bucket - turn off since bucket works only with my account for now (write access") 

args = parser.parse_args()

DAYS_SINCE_YDAY = args.days_since_yday
ECO_SYSTEM = args.eco_system.lower()
COMPUTE_DEVICE = args.compute_device.lower()
SEC_MODEL_TYPE = args.security_filter_model.lower()
CVE_MODEL_TYPE = args.probable_cve_model.lower()
S3_UPLOAD = args.s3_upload_cves

daiquiri.setup(level=logging.INFO)
_logger = daiquiri.getLogger(__name__)


# ======= BQ CLIENT SETUP FOR GETTING GITHUB BQ DATA ========
_logger.info('----- BQ CLIENT SETUP FOR GETTING GITHUB BQ DATA -----')

GH_BQ_CLIENT = bq_helper.create_github_bq_client()
if ECO_SYSTEM == 'openshift':
    REPO_NAMES = bq_helper.get_gokube_trackable_repos(repo_dir=cc.GOKUBE_REPO_LIST)
elif ECO_SYSTEM == 'knative':
    REPO_NAMES = bq_helper.get_gokube_trackable_repos(repo_dir=cc.KNATIVE_REPO_LIST)
elif ECO_SYSTEM == 'kubevirt':
    REPO_NAMES = bq_helper.get_gokube_trackable_repos(repo_dir=cc.KUBEVIRT_REPO_LIST)

_logger.info('\n')


# ======= DATES SETUP FOR GETTING GITHUB BQ DATA ========
_logger.info('----- DATES SETUP FOR GETTING GITHUB BQ DATA -----')

# Don't change this
PRESENT_TIME = arrow.now()


# CHANGE NEEDED
# to get data for N days back starting from YESTERDAY
# e.g if today is 20190528 and DURATION DAYS = 2 -> BQ will get data for 20190527, 20190526
# We don't get data for PRESENT DAY since github data will be incomplete on the same day
# But you can get it if you want but better not to for completeness :)

# You can set this directly from command line using the -d or --days-since-yday argument
DURATION_DAYS = DAYS_SINCE_YDAY or 3 # Gets 3 days of previous data including YESTERDAY


# Don't change this
# Start time for getting data
START_TIME = PRESENT_TIME.shift(days=-DURATION_DAYS)


# Don't change this
# End time for getting data (present_time - 1) i.e yesterday
# you can remove -1 to get present day data
# but it is not advised as data will be incomplete
END_TIME = PRESENT_TIME.shift(days=-1)

LAST_N_DAYS = [dt.format('YYYYMMDD')
               for dt in arrow.Arrow.range('day', START_TIME, END_TIME)]
_logger.info('Data will be retrieved for Last N={n} days: {days}'.format(n=DURATION_DAYS,
                                                                         days=LAST_N_DAYS))
_logger.info('\n')


# ======= BQ QUERY PARAMS SETUP FOR GETTING GITHUB BQ DATA ========
_logger.info('----- BQ QUERY PARAMS SETUP FOR GETTING GITHUB BQ DATA -----')

# Don't change this
YEAR_PREFIX = '20*'
DAY_LIST = [item[2:] for item in LAST_N_DAYS]
QUERY_PARAMS = {
    '{year_prefix_wildcard}': YEAR_PREFIX,
    '{year_suffix_month_day}': '(' + ', '.join(["'" + d + "'" for d in DAY_LIST]) + ')',
    '{repo_names}': '(' + ', '.join(["'" + r + "'" for r in REPO_NAMES]) + ')'
}

_logger.info('\n')


# ======= BQ GET DATASET SIZE ESTIMATE ========
_logger.info('----- BQ Dataset Size Estimate -----')

query = """
SELECT  type as EventType, count(*) as Freq
        FROM `githubarchive.day.{year_prefix_wildcard}`
        WHERE _TABLE_SUFFIX IN {year_suffix_month_day}
        AND repo.name in {repo_names}
        AND type in ('PullRequestEvent', 'IssuesEvent')
        GROUP BY type
"""
query = bq_helper.bq_add_query_params(query, QUERY_PARAMS)
df = GH_BQ_CLIENT.query_to_pandas(query)
_logger.info('Dataset Size for Last N={n} days:-'.format(n=DURATION_DAYS))
_logger.info('\n{data}'.format(data=df))

_logger.info('\n')


# ======= BQ GITHUB DATASET RETRIEVAL & PROCESSING ========
_logger.info('----- BQ GITHUB DATASET RETRIEVAL & PROCESSING -----')

ISSUE_QUERY = """
SELECT
    repo.name as repo_name,
    type as event_type,
    'golang' as ecosystem,
    JSON_EXTRACT_SCALAR(payload, '$.action') as status,
    JSON_EXTRACT_SCALAR(payload, '$.issue.id') as id,
    JSON_EXTRACT_SCALAR(payload, '$.issue.number') as number,
    JSON_EXTRACT_SCALAR(payload, '$.issue.url') as api_url,
    JSON_EXTRACT_SCALAR(payload, '$.issue.html_url') as url,
    JSON_EXTRACT_SCALAR(payload, '$.issue.user.login') as creator_name,
    JSON_EXTRACT_SCALAR(payload, '$.issue.user.html_url') as creator_url,
    JSON_EXTRACT_SCALAR(payload, '$.issue.created_at') as created_at,
    JSON_EXTRACT_SCALAR(payload, '$.issue.updated_at') as updated_at,
    JSON_EXTRACT_SCALAR(payload, '$.issue.closed_at') as closed_at,
    TRIM(REGEXP_REPLACE(
             REGEXP_REPLACE(
                 JSON_EXTRACT_SCALAR(payload, '$.issue.title'),
                 r'\\r\\n|\\r|\\n',
                 ' '),
             r'\s{2,}',
             ' ')) as title,
    TRIM(REGEXP_REPLACE(
             REGEXP_REPLACE(
                 JSON_EXTRACT_SCALAR(payload, '$.issue.body'),
                 r'\\r\\n|\\r|\\n',
                 ' '),
             r'\s{2,}',
             ' ')) as body

FROM `githubarchive.day.{year_prefix_wildcard}`
    WHERE _TABLE_SUFFIX IN {year_suffix_month_day}
    AND repo.name in {repo_names}
    AND type = 'IssuesEvent'
    """

ISSUE_QUERY = bq_helper.bq_add_query_params(ISSUE_QUERY, QUERY_PARAMS)
qsize = GH_BQ_CLIENT.estimate_query_size(ISSUE_QUERY)
_logger.info('Retrieving GH Issues. Query cost in GB={qc}'.format(qc=qsize))

issues_df = GH_BQ_CLIENT.query_to_pandas(ISSUE_QUERY)
if issues_df.empty:
    _logger.warn('No issues present for given time duration.')
else:
    _logger.info('Total issues retrieved: {n}'.format(n=len(issues_df)))

    issues_df.created_at = pd.to_datetime(issues_df.created_at)
    issues_df.updated_at = pd.to_datetime(issues_df.updated_at)
    issues_df.closed_at = pd.to_datetime(issues_df.closed_at)
    issues_df = issues_df.loc[issues_df.groupby(
        'url').updated_at.idxmax(skipna=False)].reset_index(drop=True)
    _logger.info(
        'Total issues after deduplication: {n}'.format(n=len(issues_df)))


PR_QUERY = """
SELECT
    repo.name as repo_name,
    type as event_type,
    'golang' as ecosystem,
    JSON_EXTRACT_SCALAR(payload, '$.action') as status,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.id') as id,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.number') as number,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.url') as api_url,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.html_url') as url,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.login') as creator_name,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.user.html_url') as creator_url,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.created_at') as created_at,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.updated_at') as updated_at,
    JSON_EXTRACT_SCALAR(payload, '$.pull_request.closed_at') as closed_at,
    TRIM(REGEXP_REPLACE(
             REGEXP_REPLACE(
                 JSON_EXTRACT_SCALAR(payload, '$.pull_request.title'),
                 r'\\r\\n|\\r|\\n',
                 ' '),
             r'\s{2,}',
             ' ')) as title,
    TRIM(REGEXP_REPLACE(
             REGEXP_REPLACE(
                 JSON_EXTRACT_SCALAR(payload, '$.pull_request.body'),
                 r'\\r\\n|\\r|\\n',
                 ' '),
             r'\s{2,}',
             ' ')) as body

FROM `githubarchive.day.{year_prefix_wildcard}`
    WHERE _TABLE_SUFFIX IN {year_suffix_month_day}
    AND repo.name in {repo_names}
    AND type = 'PullRequestEvent'
"""

PR_QUERY = bq_helper.bq_add_query_params(PR_QUERY, QUERY_PARAMS)
qsize = GH_BQ_CLIENT.estimate_query_size(PR_QUERY)
_logger.info(
    'Retrieving GH Pull Requests. Query cost in GB={qc}'.format(qc=qsize))

prs_df = GH_BQ_CLIENT.query_to_pandas(PR_QUERY)
if prs_df.empty:
    _logger.warn('No pull requests present for given time duration.')
else:
    _logger.info('Total pull requests retrieved: {n}'.format(n=len(prs_df)))

    prs_df.created_at = pd.to_datetime(prs_df.created_at)
    prs_df.updated_at = pd.to_datetime(prs_df.updated_at)
    prs_df.closed_at = pd.to_datetime(prs_df.closed_at)
    prs_df = prs_df.loc[prs_df.groupby('url').updated_at.idxmax(
        skipna=False)].reset_index(drop=True)
    _logger.info(
        'Total pull requests after deduplication: {n}'.format(n=len(prs_df)))

_logger.info('\n')

_logger.info('Merging issues and pull requests datasets')
cols = issues_df.columns
df = pd.concat([issues_df, prs_df], axis=0, sort=False,
               ignore_index=True).reset_index(drop=True)
df = df[cols]

df.to_csv('test_data_models.csv', index=False)

if df.empty:
    _logger.warn('Nothing to predict today :)')
else:
    _logger.info('Creating description column for NLP')
    df['description'] = df['title'].fillna(value='').map(
        str) + ' ' + df['body'].fillna(value='')
    columns = ['title', 'body']
    df.drop(columns, inplace=True, axis=1)

    if CVE_MODEL_TYPE == 'gru':
        _logger.info('Text Pre-processing Issue/PR Descriptions')
        df['norm_description'] = tn.pre_process_documents_parallel(
            documents=df['description'].values)
        df.drop(['description'], inplace=True, axis=1)

        _logger.info('Setting Default CVE and Security Flags')
        df['security_model_flag'] = 0
        df['cve_model_flag'] = 0

        _logger.info('\n')


        # ======= STARTING MODEL INFERENCE ========
        _logger.info('----- STARTING MODEL INFERENCE -----')
        _logger.info('Loading Security Model')
        sc = sdc.SecurityClassifier(embedding_size=300, max_length=1000, max_features=800000,
                                    tokenizer_path=cc.P1GRU_SEC_MODEL_TOKENIZER_PATH,
                                    model_weights_path=cc.P1GRU_SEC_MODEL_WEIGHTS_PATH)
        sc.build_model_architecture()
        sc.load_model_weights()
        sc_model = sc.get_model()

        _logger.info('Preparing data for security model inference')
        security_encoded_docs = sc.prepare_inference_data(
            df['norm_description'].tolist())
        _logger.info('Total Security Docs Encoded: {n}'.format(
            n=len(security_encoded_docs)))
        sec_doc_lengths = np.array([len(np.nonzero(item)[0])
                                    for item in security_encoded_docs])
        _logger.info('Removing bad docs with low tokens')
        sec_doc_idx = np.argwhere(sec_doc_lengths >= 5).ravel()
        filtered_security_encoded_docs = security_encoded_docs[sec_doc_idx]
        _logger.info('Filtered Security Docs Encoded: {n}'.format(
            n=len(filtered_security_encoded_docs)))

        _logger.info('Making predictions for probable security issues')
        sec_pred_probs = sc_model.predict(
            filtered_security_encoded_docs, batch_size=1024, verbose=0)
        sec_pred_probsr = sec_pred_probs.ravel()
        sec_pred_labels = [1 if prob > 0.4 else 0 for prob in sec_pred_probsr]
        _logger.info('Updating Security Model predictions in dataset')
        df.loc[df.index.isin(sec_doc_idx), 'security_model_flag'] = sec_pred_labels

        _logger.info('Teardown security model')
        del sc
        del sc_model
        gc.collect()

        _logger.info('\n')
        _logger.info('Loading CVE Model')
        cvc = cdc.CVEClassifier(embedding_size=300, max_length=1000, max_features=600000,
                                tokenizer_path=cc.P1GRU_CVE_MODEL_TOKENIZER_PATH,
                                model_weights_path=cc.P1GRU_CVE_MODEL_WEIGHTS_PATH)
        cvc.build_model_architecture()
        cvc.load_model_weights()
        cc_model = cvc.get_model()

        _logger.info('Keeping track of probable security issue rows')
        subset_df = df[df['security_model_flag'] == 1]
        prob_security_df_rowidx = np.array(subset_df.index)




        cve_encoded_docs = cvc.prepare_inference_data(
            subset_df['norm_description'].tolist())
        _logger.info('Total CVE Docs Encoded: {n}'.format(
            n=len(cve_encoded_docs)))
        cve_doc_lengths = np.array([len(np.nonzero(item)[0])
                                    for item in cve_encoded_docs])
        _logger.info('Removing bad docs with low tokens')
        cve_doc_idx = np.argwhere(cve_doc_lengths >= 10).ravel()
        filtered_cve_encoded_docs = cve_encoded_docs[cve_doc_idx]
        _logger.info('Filtered CVE Docs Encoded: {n}'.format(
            n=len(filtered_cve_encoded_docs)))

        _logger.info('Making predictions for probable CVE issues')
        cve_pred_probs = cc_model.predict(
            filtered_cve_encoded_docs, batch_size=1024, verbose=0)
        cve_pred_probsr = cve_pred_probs.ravel()
        cve_pred_labels = [1 if prob > 0.3 else 0 for prob in cve_pred_probsr]
        _logger.info('Updating CVE Model predictions in dataset')
        prob_cve_idxs = prob_security_df_rowidx[cve_doc_idx]
        df.loc[df.index.isin(prob_cve_idxs),
            'cve_model_flag'] = cve_pred_labels
        _logger.info('Teardown CVE model')
        _logger.info('\n')

        del cvc
        del cc_model
        gc.collect()


    if CVE_MODEL_TYPE == 'bert':
        aws.s3_download_folder(aws.S3_OBJ.Bucket(cc.S3_BUCKET_NAME),
                               'model_assets',
                               '/model_inference_triage_pipeline/models/')
        _logger.info('Text Pre-processing Issue/PR Descriptions')
        df['norm_description'] = tn.pre_process_documents_parallel(
            documents=df['description'].values)
        # df.drop(['description'], inplace=True, axis=1)

        _logger.info('Setting Default CVE and Security Flags')
        df['security_model_flag'] = 0
        df['cve_model_flag'] = 0

        _logger.info('\n')
        _logger.info('----- STARTING MODEL INFERENCE -----')
        _logger.info('Loading Security Model')
        sc = sdc.SecurityClassifier(embedding_size=300, max_length=1000, max_features=800000,
                                    tokenizer_path=cc.P1GRU_SEC_MODEL_TOKENIZER_PATH,
                                    model_weights_path=cc.P1GRU_SEC_MODEL_WEIGHTS_PATH)
        sc.build_model_architecture()
        sc.load_model_weights()
        sc_model = sc.get_model()

        _logger.info('Preparing data for security model inference')
        security_encoded_docs = sc.prepare_inference_data(
            df['norm_description'].tolist())
        _logger.info('Total Security Docs Encoded: {n}'.format(
            n=len(security_encoded_docs)))
        sec_doc_lengths = np.array([len(np.nonzero(item)[0])
                                    for item in security_encoded_docs])
        _logger.info('Removing bad docs with low tokens')
        sec_doc_idx = np.argwhere(sec_doc_lengths >= 5).ravel()
        filtered_security_encoded_docs = security_encoded_docs[sec_doc_idx]
        _logger.info('Filtered Security Docs Encoded: {n}'.format(
            n=len(filtered_security_encoded_docs)))

        _logger.info('Making predictions for probable security issues')
        sec_pred_probs = sc_model.predict(
            filtered_security_encoded_docs, batch_size=1024, verbose=0)
        sec_pred_probsr = sec_pred_probs.ravel()
        sec_pred_labels = [1 if prob > 0.4 else 0 for prob in sec_pred_probsr]
        _logger.info('Updating Security Model predictions in dataset')
        df.loc[df.index.isin(sec_doc_idx), 'security_model_flag'] = sec_pred_labels

        _logger.info('Teardown security model')
        del sc
        del sc_model
        gc.collect()

        _logger.info('\n')

        _logger.info('Keeping track of probable security issue rows')
        subset_df = df[df['security_model_flag'] == 1]
        prob_security_df_rowidx = np.array(subset_df.index)

        sess = tf.Session()
        BERT_MAX_SEQ_LEN = 512

        _logger.info('Loading CVE Model')
        bc = bcvec.BERTClassifier(bert_model_path=cc.BASE_BERT_UNCASED_PATH, 
                          max_seq_length=BERT_MAX_SEQ_LEN)
        bc.build_model_architecture()   

        subset_df['norm_description'] = tn.pre_process_documents_parallel_bert(
                                            documents=subset_df['description'].values) 
        cve_encoded_docs = subset_df['norm_description'].values
        _logger.info('Total CVE Docs Encoded: {n}'.format(
            n=len(cve_encoded_docs)))
        cve_doc_lengths = np.array([len(doc.split(' ')) for doc in cve_encoded_docs])
        _logger.info('Removing bad docs with low tokens')
        cve_doc_idx = np.argwhere(cve_doc_lengths >= 10).ravel()
        filtered_cve_encoded_docs = cve_encoded_docs[cve_doc_idx]
        _logger.info('Filtered CVE Docs Encoded: {n}'.format(
            n=len(filtered_cve_encoded_docs)))

        _logger.info('BERT text processing and feature engineering')
        btp_obj = btp.BertTextProcessor(tf_session=sess, 
                                  bert_model_path=cc.BASE_BERT_UNCASED_PATH, 
                                  max_seq_length=BERT_MAX_SEQ_LEN)
        btp_obj.create_bert_tokenizer()
        btp_obj.convert_text_to_input_examples(filtered_cve_encoded_docs)
        btp_obj.convert_examples_to_features()

        _logger.info('Making predictions for probable CVE issues')
        btp.initialize_vars(sess)
        bc.load_model_weights(model_weights_path=cc.P2BERT_CVE_MODEL_WEIGHTS_PATH)
        
        cve_pred_probs = bc.model_estimator.predict(x=[btp_obj.input_ids, 
                                                       btp_obj.input_masks, 
                                                       btp_obj.segment_ids],
                                                    batch_size=256,
                                                    verbose=1)
        cve_pred_probsr = cve_pred_probs.ravel()
        cve_pred_labels = [1 if prob > 0.5 else 0 for prob in cve_pred_probsr]
        from collections import Counter
        print(Counter(cve_pred_labels))
        _logger.info('Updating CVE Model predictions in dataset')
        prob_cve_idxs = prob_security_df_rowidx[cve_doc_idx]
        df.loc[df.index.isin(prob_cve_idxs),
            'cve_model_flag'] = cve_pred_labels
        _logger.info('Teardown CVE model')
        _logger.info('\n')
            
        del btp_obj
        del bc
        gc.collect()


    # ======= PREPARING PROBABLE SECURITY & CVE DATASETS ========
    _logger.info('----- PREPARING PROBABLE SECURITY & CVE DATASETS  -----')

    BASE_TRIAGE_DIR = './triaged_datasets'
    NEW_TRIAGE_SUBDIR = '-'.join([START_TIME.format('YYYYMMDD'),
                                  END_TIME.format('YYYYMMDD')])
    NEW_DIR_PATH = os.path.join(BASE_TRIAGE_DIR, NEW_TRIAGE_SUBDIR)
    if CVE_MODEL_TYPE == 'gru':
        FILE_PREFIX = 'gru_model_inference_'
    elif CVE_MODEL_TYPE == 'bert':
        FILE_PREFIX = 'bert_model_inference_'

    MODEL_INFERENCE_DATASET = os.path.join(
        NEW_DIR_PATH, FILE_PREFIX + 'full_output_' + NEW_TRIAGE_SUBDIR + '_' + ECO_SYSTEM + '.csv')
    PROBABLE_SEC_CVE_DATASET = os.path.join(
        NEW_DIR_PATH, FILE_PREFIX + 'probable_security_and_cves_' + NEW_TRIAGE_SUBDIR + '_' + ECO_SYSTEM + '.csv')
    PROBABLE_CVE_DATASET = os.path.join(
        NEW_DIR_PATH, FILE_PREFIX + 'probable_cves_' + NEW_TRIAGE_SUBDIR + '_' + ECO_SYSTEM + '.csv')
    if not os.path.exists(NEW_DIR_PATH):
        _logger.info(
            'Creating New Model Inference Directory: {}'.format(NEW_DIR_PATH))
        os.makedirs(NEW_DIR_PATH)
    else:
        _logger.info(
            'Using Existing Model Inference Directory: {}'.format(NEW_DIR_PATH))

    df.drop(['norm_description', 'description'], 
            inplace=True, errors='ignore', axis=1)
    df['triage_is_security'] = 0
    df['triage_is_cve'] = 0
    df['triage_feedback_comments'] = ''
    columns = ['ecosystem', 'repo_name', 'event_type', 'status', 'url',
               'security_model_flag', 'cve_model_flag',
               'triage_is_security', 'triage_is_cve', 'triage_feedback_comments',
               'id', 'number', 'api_url', 'created_at',
               'updated_at', 'closed_at', 'creator_name', 'creator_url']
    df = df[columns]
    _logger.info('Saving Model Inference datasets locally:'.format(
        MODEL_INFERENCE_DATASET))
    df.to_csv(MODEL_INFERENCE_DATASET, index=False)
    _logger.info('Saving Probable Security dataset:{}'.format(
        PROBABLE_SEC_CVE_DATASET))
    df[df.security_model_flag == 1].drop(['triage_is_cve'], axis=1).to_csv(
        PROBABLE_SEC_CVE_DATASET, index=False)
    _logger.info('Saving Probable CVE dataset: {}'.format(
        PROBABLE_CVE_DATASET))
    df[df.cve_model_flag == 1].drop(['triage_is_security'], axis=1).to_csv(
        PROBABLE_CVE_DATASET, index=False)
    _logger.info('\n')

    if S3_UPLOAD:
        # ======= UPLOADING INFERENCE DATASETS TO S3 BUCKET ========
        _logger.info('----- UPLOADING INFERENCE DATASETS TO S3 BUCKET  -----')
        s3_obj = aws.S3_OBJ
        bucket_name = cc.S3_BUCKET_NAME
        s3_bucket = s3_obj.Bucket(bucket_name)

        _logger.info('Uploading Saved Model Assets to S3 Bucket')
        aws.s3_upload_folder(folder_path=NEW_DIR_PATH,
                            s3_bucket_obj=s3_bucket, prefix='triaged_datasets_openshift')
    _logger.info('All done!')
