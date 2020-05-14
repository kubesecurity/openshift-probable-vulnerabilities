# This script is run on EMR. Change all paths appropriately.
import os

import arrow
import subprocess

import threading


def worker(start, end, ecosystem):
    subprocess.run(
        [
            "python3",
            "/home/hadoop/run_model_inference.py",
            "-sd",
            start.format("YYYY-MM-DD"),
            "-ed",
            end.format("YYYY-MM-DD"),
            "-eco",
            ecosystem,
            "-cve-model",
            "bert",
            "-device",
            "cpu",
            "-s3-upload",
            "True",
        ],
        env=dict(
            os.environ,
            **{
                "GOOGLE_APPLICATION_CREDENTIALS": "/home/hadoop/google-application-credentials.json",
                "BIGQUERY_CREDENTIALS_FILEPATH": "/home/hadoop/google-application-credentials.json",
                "GOKUBE_REPO_LIST": "/home/hadoop/utils/data_assets/golang-repo-list.txt",
                "KNATIVE_REPO_LIST": "/home/hadoop/utils/data_assets/knative-repo-list.txt",
                "KUBEVIRT_REPO_LIST": "/home/hadoop/utils/data_assets/kubevirt-repo-list.txt",
                "BASE_BERT_UNCASED_PATH": "/mnt2/model_assets/gokube-phase2/base_bert_tfhub_models/bert_uncased_L12_H768_A12",
                "P2BERT_CVE_MODEL_WEIGHTS_PATH": "/mnt2/model_assets/gokube-phase2/saved_models/bert_cve75_weights-ep:02-trn_loss:0.172-trn_acc:0.957-val_loss:0.164-val_acc:0.978.h5",
                "P1GRU_SEC_MODEL_TOKENIZER_PATH": "/mnt2/model_assets/gokube-phase1-jun19/embeddings/security_tokenizer_word2idx_fulldata.pkl",
                "P1GRU_SEC_MODEL_WEIGHTS_PATH": "/mnt2/model_assets/gokube-phase1-jun19/saved_models/security_model_train99-jun19_weights.h5",
                "P1GRU_CVE_MODEL_TOKENIZER_PATH": "/mnt2/model_assets/gokube-phase1-jun19/embeddings/cve_tokenizer_word2idx_fulldata.pkl",
                "P1GRU_CVE_MODEL_WEIGHTS_PATH": "/mnt2/model_assets/gokube-phase1-jun19/saved_models/cve_model_train99-jun19_weights.h5",
                "S3_MODEL_REFRESH": "False",
                "BASE_TRIAGE_DIR": "/mnt1/triaged_datasets",
                "P2_PYTORCH_CVE_BERT_CLASSIFIER_PATH": "/mnt2/pytorch-cve-warmup",
            }
        ),
    )


def main():
    for ecosystem in ["openshift", "kubevirt", "knative"]:
        start = arrow.get("01-07-2019", "DD-MM-YYYY")
        end = start.shift(days=+6)
        end_final = arrow.get("12-04-2020", "DD-MM-YYYY")

        pool = []
        while end <= end_final:
            print(
                "Creating for {} to {}".format(start.format("YYYY-MM-DD"), end.format("YYYY-MM-DD"))
            )
            pool.append(threading.Thread(target=worker, args=(start, end, ecosystem)))
            if len(pool) >= 16:
                for thread in pool:
                    thread.start()
                for thread in pool:
                    thread.join()
                # Wait for all 16 to execute
                pool = []
            start = end.shift(days=+1)
            end = start.shift(days=+6)
        if len(pool) != 0:
            for thread in pool:
                thread.start()
            for thread in pool:
                thread.join()


if __name__ == "__main__":
    main()
