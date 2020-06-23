"""Functions for text processing and feature engineering for BERT models."""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # noqa
import bert
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
from tensorflow.keras import backend as K
from tqdm import tqdm


def initialize_vars(tf_session):
    """Initialize tf session variables."""
    tf_session.run(tf.local_variables_initializer())
    tf_session.run(tf.global_variables_initializer())
    tf_session.run(tf.tables_initializer())
    K.set_session(tf_session)


class PaddingInputExample:
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size.

    The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    batches could cause silent errors.

    Won't usually cause issues on CPU/GPU hopefully.
    """


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Construct InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence.
            For single sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second
            sequence. Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.

        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class BertTextProcessor:
    """Text processor related to BERT."""

    def __init__(self, tf_session, bert_model_path, max_seq_length=128):
        """Construct BertTextProcessor object."""
        self.tokenizer = None
        self.bert_path = bert_model_path
        self.tf_sess = tf_session
        self.input_examples = []
        self.max_seq_length = max_seq_length
        self.input_ids = None
        self.input_masks = None
        self.segment_ids = None
        self.labels = None

    def create_bert_tokenizer(self):
        """Get vocab file and casing info from BERT tensorflow hub model."""
        print('Loading Base BERT Model')
        bert_model = tf_hub.Module(self.bert_path)
        tokenization_info = bert_model(signature="tokenization_info", as_dict=True)
        vocab_file, do_lower_case = self.tf_sess.run(
            [
                tokenization_info["vocab_file"],
                tokenization_info["do_lower_case"],
            ]
        )
        print('Loading BERT WordPiece Tokenizer')
        self.tokenizer = bert.tokenization.FullTokenizer(vocab_file=vocab_file,
                                                         do_lower_case=do_lower_case)

    def convert_text_to_input_examples(self, texts, labels=[]):
        """Create InputExamples.

        It is based on instances of the
        bert.run_classifier.InputExample class.
        """
        labels = labels or [None] * len(texts)
        print('Creating Input Examples from data')
        for text, label in tqdm(zip(texts, labels),
                                desc="Converting text to examples"):
            self.input_examples.append(
                InputExample(guid=None, text_a=text, text_b=None, label=label)
            )

    def convert_single_example(self, tokenizer, example, max_seq_length):
        """Convert a single example instance of class InputExample.

        into a single instance of features which consist of the following
            - input_id
            - input_mask
            - segment_id
            - label (None in case of inference)
        this is based on instances of the bert.run_classifier.InputFeatures
        class which is usually generated from the function
        bert.run_classifier.convert_single_example()
        """
        if isinstance(example, PaddingInputExample):
            input_ids = [0] * max_seq_length
            input_mask = [0] * max_seq_length
            segment_ids = [0] * max_seq_length
            label = 0
            return input_ids, input_mask, segment_ids, label

        tokens_a = tokenizer.tokenize(example.text_a)
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens.
        # Only real tokens are attended to in the attention layers.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # double check lengths are alright
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return input_ids, input_mask, segment_ids, example.label

    def convert_examples_to_features(self):
        """Convert a set of `InputExample` instancess to a list.

        of instances of`InputFeatures` using the
        convert_single_example(...) function.
        """
        print('Creating BERT Input Features from Input Examples')
        input_ids, input_masks, segment_ids, labels = [], [], [], []
        for example in tqdm(self.input_examples,
                            desc="Converting examples to features"):
            input_id, input_mask, segment_id, label = self.convert_single_example(
                self.tokenizer, example, self.max_seq_length
            )
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels.append(label)

        self.input_ids = np.array(input_ids)
        self.input_masks = np.array(input_masks)
        self.segment_ids = np.array(segment_ids)
        self.labels = np.array(labels).reshape(-1, 1)
