"""Trains the BERT-Tensorflow BERT model (Uses Tensorflow 1.x). Legacy in go-cve context."""
import logging
from collections import Counter

import daiquiri
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from models import bert_cve_classifier as bcvec
from utils import bert_text_processor as btp
from utils import cloud_constants as cc
from utils import model_constants as mc
from pathlib import Path

daiquiri.setup(level=logging.DEBUG)
_logger = daiquiri.getLogger(__name__)

SEED = 42
np.random.seed(SEED)
tf.set_random_seed(SEED)

home = Path().home()

# Ensure that data is downloaded from S3 in the EMR bootstrap scripts.
dataset = pd.read_csv(
    home.joinpath("GH_complete_labeled_issues_prs - preprocessed.csv").as_posix(),
    encoding="utf-8",
    na_filter=False,
)
dataset = dataset[dataset.label != 0]
texts = dataset["description"].tolist()
labels = dataset["label"].tolist()

_logger.debug("Before: {}".format(Counter(labels)))
labels = [0 if item == 1 else 1 for item in labels]
_logger.debug("After: {}".format(Counter(labels)))


train_text, test_text, train_labels, test_labels = train_test_split(
    texts, labels, test_size=1e-4, random_state=SEED
)
_logger.debug("Before:", len(train_text), len(test_text))


# Initialize session
sess = tf.Session()

# Params for bert model and tokenization
BERT_PATH = cc.BASE_BERT_UNCASED_PATH
MAX_SEQ_LENGTH = mc.MAX_SEQ_LENGTH


train_text_lengths = np.array([len(doc.split(" ")) for doc in train_text])

train_text_idx = np.argwhere(train_text_lengths >= 5).ravel()
train_text = [train_text[i] for i in train_text_idx]
train_labels = [train_labels[i] for i in train_text_idx]
_logger.debug("After: {}, {}".format(len(train_text), len(train_labels)))

# process text data
btp_train = btp.BertTextProcessor(
    tf_session=sess, bert_model_path=BERT_PATH, max_seq_length=MAX_SEQ_LENGTH
)
btp_train.create_bert_tokenizer()
btp_train.convert_text_to_input_examples(train_text, train_labels)
btp_train.convert_examples_to_features()

# load pre-trained base BERT model
bc = bcvec.BERTClassifier(bert_model_path=BERT_PATH, max_seq_length=MAX_SEQ_LENGTH)
bc.build_model_architecture()
_logger.debug(bc.model_estimator.summary())

class_weights = class_weight.compute_class_weight(
    "balanced", np.unique(btp_train.labels.ravel()), btp_train.labels.ravel()
)
class_weights = dict(enumerate(class_weights))
class_weights[1] *= 2
_logger.debug(class_weights)

modelckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    "bert_cve99iter3_weights-ep:{epoch:02d}-trn_loss:{loss:.3f}-trn_acc:{acc:.3f}-val_loss:{val_loss:.3f}-val_acc:{val_acc:.3f}.h5",
    save_weights_only=True,
    period=1,
    verbose=1,
)
btp.initialize_vars(sess)
history = bc.model_estimator.fit(
    x=[btp_train.input_ids, btp_train.input_masks, btp_train.segment_ids],
    y=train_labels,
    validation_split=0.1,
    epochs=10,
    batch_size=15,
    class_weight=class_weights,
    callbacks=[modelckpt_cb],
    verbose=1,
)

_logger.debug(history.history)
