import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import bert_text_processor as btp
from models import bert_cve_classifier as bcvec
from sklearn.utils import class_weight
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

SEED = 42
np.random.seed(SEED)
tf.set_random_seed(SEED)



dataset = pd.read_csv('../../data/GH_complete_labeled_issues_prs - preprocessed.csv', encoding='utf-8', 
                      na_filter=False)
dataset = dataset[dataset.label != 0]
texts = dataset['description'].tolist()
labels = dataset['label'].tolist()

print('Before:', Counter(labels))
labels = [0 if item == 1 else 1 for item in labels]
print('After:', Counter(labels))


train_text, test_text, train_labels, test_labels = train_test_split(texts, labels, 
                                                                    test_size=1e-4, random_state=SEED)
print('Before:', len(train_text), len(test_text))


# Initialize session
sess = tf.Session()

# Params for bert model and tokenization
BERT_PATH = "models/model_assets/gokube-phase2/base_bert_tfhub_models/bert_uncased_L12_H768_A12"
MAX_SEQ_LENGTH = 512


train_text_lengths = np.array([len(doc.split(' ')) for doc in train_text])

train_text_idx = np.argwhere(train_text_lengths >= 5).ravel()
train_text = [train_text[i] for i in train_text_idx]
train_labels = [train_labels[i] for i in train_text_idx]
print('After:', len(train_text), len(train_labels))

# process text data
btp_train = btp.BertTextProcessor(tf_session=sess, 
                                  bert_model_path=BERT_PATH, 
                                  max_seq_length=MAX_SEQ_LENGTH)
btp_train.create_bert_tokenizer()
btp_train.convert_text_to_input_examples(train_text, train_labels)
btp_train.convert_examples_to_features()

# load pre-trained base BERT model
bc = bcvec.BERTClassifier(bert_model_path=BERT_PATH, 
                          max_seq_length=MAX_SEQ_LENGTH)
bc.build_model_architecture()
print(bc.model_estimator.summary())

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(btp_train.labels.ravel()),
                                                  btp_train.labels.ravel())
class_weights = dict(enumerate(class_weights))
class_weights[1] *= 2
print(class_weights)

modelckpt_cb = tf.keras.callbacks.ModelCheckpoint('bert_cve99iter3_weights-ep:{epoch:02d}-trn_loss:{loss:.3f}-trn_acc:{acc:.3f}-val_loss:{val_loss:.3f}-val_acc:{val_acc:.3f}.h5', 
                                        save_weights_only=True, period=1, verbose=1)
btp.initialize_vars(sess)
history = bc.model_estimator.fit(x=[btp_train.input_ids, 
                                    btp_train.input_masks, 
                                    btp_train.segment_ids],
                                 y=train_labels,
                                 validation_split=0.1,
                                 epochs=10,
                                 batch_size=15,
                                 class_weight=class_weights,
                                 callbacks=[modelckpt_cb],
                                 verbose=1
)

print(history.history)