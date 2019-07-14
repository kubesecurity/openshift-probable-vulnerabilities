# GoKube-OpenShift Probable Vulnerability Inference Pipeline

This directory has all the scripts and models which are used to leverage our trained deep learning models to predict probable CVE and Security issues on approximately 850 repos in the Kube\Go\Openshift eco-system. Anyone can leverage our pre-trained AI models and perform inference on live GitHub events data to predict probable vulnerabilities using this codebase. You would also need the right auth credentials for the same.

<br>

## Current AI Models for Inference
We have a total of two AI models in our inference pipeline. These include,
  1. A deep learning model to filter out non-security related GitHub events.
  2. A deep learning model which works on the output of model (1) and predicts probable vulnerabilities.

The following figure depicts our model inference and triaging pipeline in further detail.
![](https://i.imgur.com/LU0V3gN.png)
Right now, we are using a stacked GRU + Attention-based deep learning model for our two models mentioned above. We are also looking at transformer architectures like BERT to be used in the future.

<br>

## Model Inference and Triage Instructional Video

This screencast showcases a step-by-step procedure to leverage our pre-trained AI models and perform model inference and triage of probable vulnerabilities.

[![asciicast](https://asciinema.org/a/8gA9nviECAmOaoyLObkEoaEPr.svg)](https://asciinema.org/a/8gA9nviECAmOaoyLObkEoaEPr)

<br>

## Model Inference and Triage Step-by-Step Instructions

If you'd rather want to follow a step-by-step process without watching the video, this section is just for you :relaxed:

#### Step 1: Go to the model inference directory
Make sure you are at the right directory for running the model inference pipeline.
```bash
[dsarkar@localhost openshift-probable-vulnerabilities]$ # go to model inference dir 
[dsarkar@localhost openshift-probable-vulnerabilities]$ cd model_inference_triage_pipeline/   
[dsarkar@localhost model_inference_triage_pipeline]$ ls -l                          
total 52                                                                           
drwxr-xr-x. 2 dsarkar dsarkar  4096 Jul 12 14:17 models                             
drwxrwxr-x. 2 dsarkar dsarkar  4096 May 29 17:03 notebooks                           
-rw-rw-r--. 1 dsarkar dsarkar   508 May 29 17:03 prepare_inference_envt.py           
-rw-rw-r--. 1 dsarkar dsarkar   414 Jul 11 11:00 README.md                           
-rw-rw-r--. 1 dsarkar dsarkar   135 May 29 17:03 requirements.in                    
-rw-rw-r--. 1 dsarkar dsarkar  2266 May 29 17:03 requirements.txt                   
-rw-rw-r--. 1 dsarkar dsarkar 15051 May 29 17:03 run_model_inference.py             
-rw-rw-r--. 1 dsarkar dsarkar   519 May 29 17:03 upload_model_assets.py             
-rw-rw-r--. 1 dsarkar dsarkar  1296 May 29 17:03 upload_triage_feedback.py          
drwxrwxr-x. 4 dsarkar dsarkar  4096 May 29 17:06 utils 
```
<br>

#### Step 2: Create and activate a python virtual environment
We will create an isolated python virtual environment to run our code to prevent any conflicting installs with our global python installation.
```bash
[dsarkar@localhost model_inference_triage_pipeline]$ virtualenv venv                 
Using base prefix '/home/dsarkar/anaconda3'                         
New python executable in /home/dsarkar/repos/openshift-probable-vulnerabilities/model_inference_triage_pipeline/venv/bin/python 
Installing setuptools, pip, wheel...                                                
done.  

[dsarkar@localhost model_inference_triage_pipeline]$ source venv/bin/activate        
(venv) [dsarkar@localhost model_inference_triage_pipeline]$ which python3           
~/repos/openshift-probable-vulnerabilities/model_inference_triage_pipeline/venv/bin/python3  
```
<br>

#### Step 3: Install necessary dependencies
We install the necessary dependencies which are used by our inference system, particularly our deep learning models and file transfer to S3.
```bash
(venv) [dsarkar@localhost model_inference_triage_pipeline]$ pip install -r requirements.txt
Collecting absl-py==0.7.1 (from -r requirements.txt (line 7))                       
Collecting arrow==0.13.2 (from -r requirements.txt (line 8))                         
  Using cached https://files.pythonhosted.org/packages/28/2f/1be1d6914409d27a3eefc621676a50951edafca30f74bd731c8fb5ecdc24/arrow-0.13.2-py2.py3-none-any.whl                                                        
Collecting astor==0.8.0 (from -r requirements.txt (line 9))                         
  Using cached https://files.pythonhosted.org/packages/d1/4f/950dfae467b384fc96bc6469de25d832534f6b4441033c39f914efd13418/astor-0.8.0-py2.py3-none-any.whl                                                         
Collecting beautifulsoup4==4.7.1 (from -r requirements.txt (line 10))               
  Using cached https://files.pythonhosted.org/packages/1d/5d/3260694a59df0ec52f8b4883f5d23b130bc237602a1411fa670eae12351e/beautifulsoup4-4.7.1-py3-none-any.whl                                                    
Collecting boto3==1.9.157 (from -r requirements.txt (line 11)) 
...
...
Collecting tensorflow==1.12.0 (from -r requirements.txt (line 51))                   
  Using cached https://files.pythonhosted.org/packages/22/cc/ca70b78087015d21c5f3f93694107f34ebccb3be9624385a911d4b52ecef/tensorflow-1.12.0-cp36-cp36m-manylinux1_x86_64.whl                                       
Collecting termcolor==1.1.0 (from -r requirements.txt (line 52))                      
Collecting urllib3==1.25.3 (from -r requirements.txt (line 53))                      
  Using cached https://files.pythonhosted.org/packages/e6/60/247f23a7121ae632d62811ba7f273d0e58972d75e58a94d329d51550a47d/urllib3-1.25.3-py2.py3-none-any.whl                                                      
Collecting werkzeug==0.15.4 (from -r requirements.txt (line 54))                     
  Using cached https://files.pythonhosted.org/packages/9f/57/92a497e38161ce40606c27a86759c6b92dd34fcdb33f64171ec559257c02/Werkzeug-0.15.4-py2.py3-none-any.whl                                                     
Requirement already satisfied: wheel==0.33.4 in ./venv/lib/python3.6/site-packages (from -r requirements.txt (line 55)) (0.33.4)                                                                                   
Requirement already satisfied: setuptools>=34.0.0 in ./venv/lib/python3.6/site-packages (from google-api-core==1.11.1->-r requirements.txt (line 21)) (41.0.1)                                                     
Installing collected packages: six, absl-py, python-dateutil, arrow, astor, soupsieve, beautifulsoup4, jmespath, urllib3, docutils, botocore, s3transfer, boto3, cachetools, certifi, chardet, contractions, daiqui
ri, dill, gast, protobuf, googleapis-common-protos, pyasn1, pyasn1-modules, rsa, google-auth, pytz, idna, requests, google-api-core, google-cloud-core, google-resumable-media, google-cloud-bigquery, grpcio, nump
y, h5py, keras-applications, keras-preprocessing, pyyaml, scipy, keras, lxml, markdown, pandas, werkzeug, tensorboard, termcolor, tensorflow                                                                       
Successfully installed absl-py-0.7.1 arrow-0.13.2 astor-0.8.0 beautifulsoup4-4.7.1 boto3-1.9.157 botocore-1.12.157 cachetools-3.1.1 certifi-2019.3.9 chardet-3.0.4 contractions-0.0.18 daiquiri-1.5.0 dill-0.2.9 do
cutils-0.14 gast-0.2.2 google-api-core-1.11.1 google-auth-1.6.3 google-cloud-bigquery-1.12.1 google-cloud-core-1.0.0 google-resumable-media-0.3.2 googleapis-common-protos-1.6.0 grpcio-1.21.1 ... tensorboard-1.12.2 tensorflow-1.12.0 termcolor-1.1.0 urllib3-1.25.3 werkzeug-0.15.4  
```
<br>

#### Step 4: Download pre-trained AI model assets
We have our pre-trained AI model assets (weights, vocabulary) stored in the cloud on S3. To run model inference, we first need to download these assets locally considering they are pretty big. Download time depends on the speed of your internet.
```bash
(venv) [dsarkar@localhost model_inference_triage_pipeline]$ python3 prepare_inference_envt.py   
2019-07-12 14:21:23,226 [12823] INFO     __main__: Downloading Saved Model Assets from S3 Bucket  
2019-07-12 14:21:24,512 [12823] INFO     utils.aws_utils: Downloading object: model_assets/gokube-phase1-jun19/embeddings/cve_tokenizer_word2idx_fulldata.pkl                                                      
2019-07-12 14:21:45,508 [12823] INFO     utils.aws_utils: Downloading object: model_assets/gokube-phase1-jun19/embeddings/security_tokenizer_word2idx_fulldata.pkl                                                 
2019-07-12 14:22:03,239 [12823] INFO     utils.aws_utils: Downloading object: model_assets/gokube-phase1-jun19/saved_models/cve_model_train99-jun19_weights.h5                                                     
2019-07-12 14:23:26,201 [12823] INFO     utils.aws_utils: Downloading object: model_assets/gokube-phase1-jun19/saved_models/security_model_train99-jun19_weights.h5                                                
(venv) [dsarkar@localhost model_inference_triage_pipeline]$ # check saved model assets 
(venv) [dsarkar@localhost model_inference_triage_pipeline]$ tree models/             
models/                                                                             
├── cve_dl_classifier.py                                                            
├── __init__.py                                                                      
├── model_assets                                                                     
│   └── gokube-phase1-jun19                                                          
│       ├── embeddings                                                               
│       │   ├── cve_tokenizer_word2idx_fulldata.pkl                                    
│       │   └── security_tokenizer_word2idx_fulldata.pkl                              
│       └── saved_models                                                             
│           ├── cve_model_train99-jun19_weights.h5                                    
│           └── security_model_train99-jun19_weights.h5                               
└── security_dl_classifier.py                                                        
4 directories, 7 files
```
<br>

#### Step 5: Run model inference pipeline
There is already a script which helps run the entire model inference pipeline to predict probable vulnerabilities. The usage details are depicted as follows.
```bash
(venv) [dsarkar@localhost model_inference_triage_pipeline]$ python3 run_model_inference.py --help       
Using TensorFlow backend.                                                            
usage: run_model_inference.py [-h] [-d DAYS_SINCE_YDAY] 

optional arguments:                                                                  
  -h, --help            show this help message and exit                              
  -d DAYS_SINCE_YDAY, --days-since-yday DAYS_SINCE_YDAY                              
                        The number of days worth of data to retrieve from             
                        GitHub including yesterday                                  
(venv) [dsarkar@localhost model_inference_triage_pipeline]$ # this takes one argument -d for number of days we want to go back since day before today 
```

But before we run this, we need to make sure our authentication credentials are all setup in the [`cloud_constants.py`](https://github.com/fabric8-analytics/openshift-probable-vulnerabilities/blob/master/model_inference_triage_pipeline/utils/cloud_constants.py) file, part of which is depicted below.
```bash
# Please make sure you have your AWS envt variables setup                             
AWS_S3_REGION = os.environ.get('AWS_S3_REGION', 'us-east-1')                         
AWS_S3_ACCESS_KEY_ID = os.environ.get('AWS_S3_ACCESS_KEY_ID', '')                     
AWS_S3_SECRET_ACCESS_KEY = os.environ.get('AWS_S3_SECRET_ACCESS_KEY', '')  
...
... 
# Please set the following to point to your BQ auth credentials JSON                 
BIGQUERY_CREDENTIALS_FILEPATH = '../../auth/bq_key.json'                
...
```

Once your AWS and Big Query credentials are setup, its time to run the model inference pipeline!
```bash
(venv) [dsarkar@localhost model_inference_triage_pipeline]$ date                     
Fri Jul 12 14:26:51 IST 2019                                                        
(venv) [dsarkar@localhost model_inference_triage_pipeline]$ # lets run it for a week back from 11th July   
(venv) [dsarkar@localhost model_inference_triage_pipeline]$ python3 run_model_inference.py -d 7  
Using TensorFlow backend.                                                             
2019-07-12 14:28:03,184 [13836] INFO     __main__: ----- BQ CLIENT SETUP FOR GETTING GITHUB BQ DATA -----  
2019-07-12 14:28:03,187 [13836] INFO     utils.bq_client_helper: Setting up BQ Client: <utils.bq_utils.BigQueryHelper object at 0x7f25d821e748>                                                                    
2019-07-12 14:28:03,190 [13836] INFO     utils.bq_client_helper: Total Repos to Track: 845    
2019-07-12 14:28:03,190 [13836] INFO     __main__:                                 
2019-07-12 14:28:03,190 [13836] INFO     __main__: ----- DATES SETUP FOR GETTING GITHUB BQ DATA -----  
2019-07-12 14:28:03,191 [13836] INFO     __main__: Data will be retrieved for Last N=7 days: ['20190705', '20190706', '20190707', '20190708', '20190709', '20190710', '20190711']                                  
2019-07-12 14:28:03,191 [13836] INFO     __main__:                                
2019-07-12 14:28:03,191 [13836] INFO     __main__: ----- BQ QUERY PARAMS SETUP FOR GETTING GITHUB BQ DATA -----   
2019-07-12 14:28:03,191 [13836] INFO     __main__:                                
2019-07-12 14:28:03,191 [13836] INFO     __main__: ----- BQ Dataset Size Estimate -----     
2019-07-12 14:28:19,553 [13836] INFO     __main__: Dataset Size for Last N=7 days:-      
2019-07-12 14:28:19,575 [13836] INFO     __main__:                                   
          EventType  Freq                                                           
0  PullRequestEvent  2657                                                           
1       IssuesEvent  1558                                                            
2019-07-12 14:28:19,575 [13836] INFO     __main__:                                
2019-07-12 14:28:19,575 [13836] INFO     __main__: ----- BQ GITHUB DATASET RETRIEVAL & PROCESSING -----  
2019-07-12 14:28:25,790 [13836] INFO     __main__: Retrieving GH Issues. Query cost in GB=30.528669455088675 
2019-07-12 14:28:39,511 [13836] INFO     __main__: Total issues retrieved: 1558          
2019-07-12 14:28:39,739 [13836] INFO     __main__: Total issues after deduplication: 1327 
2019-07-12 14:28:44,631 [13836] INFO     __main__: Retrieving GH Pull Requests. Query cost in GB=30.528669455088675   
2019-07-12 14:28:57,387 [13836] INFO     __main__: Total pull requests retrieved: 2657 
2019-07-12 14:28:57,699 [13836] INFO     __main__: Total pull requests after deduplication: 1873  
2019-07-12 14:28:57,700 [13836] INFO     __main__:       
2019-07-12 14:28:57,700 [13836] INFO     __main__: Merging issues and pull requests datasets  
2019-07-12 14:28:57,706 [13836] INFO     __main__: Creating description column for NLP  
2019-07-12 14:28:57,715 [13836] INFO     __main__: Text Pre-processing Issue/PR Descriptions  
2019-07-12 14:28:57,716 [13836] INFO     utils.text_normalizer: Text Pre-processing: starting 
2019-07-12 14:28:57,717 [13836] INFO     utils.text_normalizer: ThreadPoolExecutor-0_0: working on doc num: 0 
2019-07-12 14:29:04,294 [13836] INFO     utils.text_normalizer: ThreadPoolExecutor-0_4: working on doc num: 3199 
2019-07-12 14:29:04,339 [13836] INFO     __main__: Setting Default CVE and Security Flags  
2019-07-12 14:29:04,341 [13836] INFO     __main__:                             
2019-07-12 14:29:04,341 [13836] INFO     __main__: ----- STARTING MODEL INFERENCE -----   
2019-07-12 14:29:04,341 [13836] INFO     __main__: Loading Security Model           
2019-07-12 14:29:04,341 [13836] INFO     models.security_dl_classifier: Loading Security Model Tokenizer Vocabulary
2019-07-12 14:29:04,532 [13836] INFO     models.security_dl_classifier: Building Security Model Architecture  
2019-07-12 14:29:05,042 [13836] INFO     models.security_dl_classifier: Loading Security Model Weights  
2019-07-12 14:29:06,781 [13836] INFO     __main__: Preparing data for security model inference  
2019-07-12 14:29:07,010 [13836] INFO     __main__: Total Security Docs Encoded: 3200       
2019-07-12 14:29:07,027 [13836] INFO     __main__: Removing bad docs with low tokens  
2019-07-12 14:29:07,032 [13836] INFO     __main__: Filtered Security Docs Encoded: 3146      
2019-07-12 14:29:07,032 [13836] INFO     __main__: Making predictions for probable security issues 
2019-07-12 14:30:05,170 [13836] INFO     __main__: Updating Security Model predictions in dataset   
2019-07-12 14:30:05,174 [13836] INFO     __main__: Teardown security model          
2019-07-12 14:30:05,235 [13836] INFO     __main__:                               
2019-07-12 14:30:05,236 [13836] INFO     __main__: Loading CVE Model                
2019-07-12 14:30:05,236 [13836] INFO     models.cve_dl_classifier: Loading CVE Model Tokenizer Vocabulary    
2019-07-12 14:30:05,382 [13836] INFO     models.cve_dl_classifier: Building CVE Model Architecture    
2019-07-12 14:30:06,071 [13836] INFO     models.cve_dl_classifier: Loading CVE Model Weights  
2019-07-12 14:30:07,771 [13836] INFO     __main__: Keeping track of probable security issue rows 
2019-07-12 14:30:07,839 [13836] INFO     __main__: Total CVE Docs Encoded: 483        
2019-07-12 14:30:07,842 [13836] INFO     __main__: Removing bad docs with low tokens   
2019-07-12 14:30:07,842 [13836] INFO     __main__: Filtered CVE Docs Encoded: 475     
2019-07-12 14:30:07,842 [13836] INFO     __main__: Making predictions for probable CVE issues  
2019-07-12 14:30:16,757 [13836] INFO     __main__: Updating CVE Model predictions in dataset  
2019-07-12 14:30:16,759 [13836] INFO     __main__:                              
2019-07-12 14:30:16,759 [13836] INFO     __main__: Teardown CVE model                 
2019-07-12 14:30:16,816 [13836] INFO     __main__: ----- PREPARING PROBABLE SECURITY & CVE DATASETS  -----  
2019-07-12 14:30:16,817 [13836] INFO     __main__: Creating New Model Inference Directory: ./triaged_datasets/20190705-20190711                                                                                    
2019-07-12 14:30:16,826 [13836] INFO     __main__: Saving Model Inference datasets locally:  
2019-07-12 14:30:16,907 [13836] INFO     __main__: Saving Probable Security dataset:./triaged_datasets/20190705-20190711/probable_security_and_cves_20190705-20190711.csv                                          
2019-07-12 14:30:16,925 [13836] INFO     __main__: Saving Probable CVE dataset: ./triaged_datasets/20190705-20190711/probable_cves_20190705-20190711.csv                                                           
2019-07-12 14:30:16,930 [13836] INFO     __main__:                                 
2019-07-12 14:30:16,930 [13836] INFO     __main__: ----- UPLOADING INFERENCE DATASETS TO S3 BUCKET  ----- 
2019-07-12 14:30:16,932 [13836] INFO     __main__: Uploading Saved Model Assets to S3 Bucket   
2019-07-12 14:30:16,932 [13836] INFO     utils.aws_utils: Uploading to: triaged_datasets/20190705-20190711/probable_cves_20190705-20190711.csv                                                                     
2019-07-12 14:30:18,533 [13836] INFO     utils.aws_utils: Uploading to: triaged_datasets/20190705-20190711/model_inference_full_output_20190705-20190711.csv                                                       
2019-07-12 14:30:23,549 [13836] INFO     utils.aws_utils: Uploading to: triaged_datasets/20190705-20190711/probable_security_and_cves_20190705-20190711.csv                                                        
2019-07-12 14:30:24,471 [13836] INFO     __main__: All done!  
```
<br>

#### Step 6: Triage probable vulnerabilities
Once model inference is complete, it usually generates three files as depicted below.
```bash
(venv) [dsarkar@localhost model_inference_triage_pipeline]$ cd triaged_datasets/    
(venv) [dsarkar@localhost triaged_datasets]$ tree 20190705-20190711/                 
20190705-20190711/                                                                  
├── model_inference_full_output_20190705-20190711.csv                                
├── probable_cves_20190705-20190711.csv                                               
└── probable_security_and_cves_20190705-20190711.csv                               
0 directories, 3 files                                                                                                                                                                                             
(venv) [dsarkar@localhost triaged_datasets]$ # ideally you want to work with the probable CVEs file and triage that  
(venv) [dsarkar@localhost triaged_datasets]$ cat 20190705-20190711/probable_cves_20190705-20190711.csv   
ecosystem,repo_name,event_type,status,url,security_model_flag,cve_model_flag,triage_is_cve,triage_feedback_comments,id,number,api_url,created_at,updated_at,closed_at,creator_name,creator_url                     
golang,golang/go,IssuesEvent,opened,https://github.com/golang/go/issues/33026,1,1,0,,466376417,33026,https://api.github.com/repos/golang/go/issues/33026,2019-07-10 15:28:31+00:00,2019-07-10 15:28:31+00:00,,Micha
elTJones,https://github.com/MichaelTJones                                             
golang,istio/istio,IssuesEvent,closed,https://github.com/istio/istio/issues/14220,1,1,0,,445632966,14220,https://api.github.com/repos/istio/istio/issues/14220,2019-05-17 21:58:47+00:00,2019-07-08 06:40:55+00:00,
2019-07-08 06:40:55+00:00,kramvan1,https://github.com/kramvan1        
...
```
Thus as mentioned, you would typically want to work on the `probable_cves_*.csv` file and triage it. The fields of interest are depicted in the following snapshot.
- __`triage_is_cve`__: You add in a __1__ if the model prediction looks to be valid else keep it as a __0__
- __`triage_feedback_comments`__: You add in any comments which might help the model improve in the future

![](https://i.imgur.com/1V3Ykvq.png)

<br>

#### Step 7: Record and send feedback 
Once initial triaging is done, you can optionally update the saved feedback to S3 using the following commands.
```bash
(venv) [dsarkar@localhost model_inference_triage_pipeline]$ python upload_triage_feedback.py --help 
usage: upload_triage_feedback.py [-h] [-ts TRIAGE_SUBFOLDER]                       
optional arguments:                                                                   
  -h, --help            show this help message and exit                               
  -ts TRIAGE_SUBFOLDER, --triage-subfolder TRIAGE_SUBFOLDER                           
                        The specific triage sub-folder to upload to S3. Folder        
                        format is YYYYMMDD-YYYYMMDD.                                                                                                                                                               
(venv) [dsarkar@localhost model_inference_triage_pipeline]$ python3 upload_triage_feedback.py -ts 20190705-20190711 
2019-07-12 14:33:23,036 [14347] INFO     __main__: Uploading Saved Model Assets to S3 Bucket 
2019-07-12 14:33:23,037 [14347] INFO     utils.aws_utils: Uploading to: triaged_datasets/20190705-20190711/probable_cves_20190705-20190711.csv                                                                     
2019-07-12 14:33:24,637 [14347] INFO     utils.aws_utils: Uploading to: triaged_datasets/20190705-20190711/model_inference_full_output_20190705-20190711.csv                                                       
2019-07-12 14:33:29,404 [14347] INFO     utils.aws_utils: Uploading to: triaged_datasets/20190705-20190711/probable_security_and_cves_20190705-20190711.csv
```

Also the reports are then verified and updated in the relevant spreadsheets in the [__`probable-vulnerabilities`__](https://github.com/dipanjanS/openshift-probable-vulnerabilities/tree/djsarkar-dev/probable-vulnerabilities) directory.












[Setup and Triage Instructions](https://drive.google.com/a/redhat.com/file/d/1IMw8qH9H_JY3jvgU8P4a11p8eDsYY3cF/view?usp=sharing)
