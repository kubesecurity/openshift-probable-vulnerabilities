import numpy as np
import tensorflow as tf
import keras
import dill
from keras.engine.topology import Layer
from keras import backend as K


# for reproducibility
SEED = 42
np.random.seed(SEED)
tf.set_random_seed(SEED)


class AttentionLayer(Layer):

    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        """

        self.supports_masking = True
        self.init = keras.initializers.get('glorot_uniform')

        self.W_regularizer = keras.regularizers.get(W_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)

        self.W_constraint = keras.constraints.get(W_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # old code doesn't work
        # eij = K.dot(x, self.W) TF backend doesn't support it
        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))),
                        (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

    def get_config(self):
        config = {'step_dim': self.step_dim}
        base_config = super(AttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ModelNotBuiltException(Exception):
    pass


class SecurityClassifier:

    def __init__(self, embedding_size=300, max_features=None,
                 max_length=None, tokenizer_path=None):

        self.EMBED_SIZE = 300 if not embedding_size else embedding_size
        self.MAX_FEATURES = 800000 if not max_features else max_features
        self.MAX_LEN = 1000 if not max_length else max_length
        self.TOKENIZER_PATH = '../../../../tokenizer_vocab/sec_tokenizer_word2idx.pkl' \
                                    if not tokenizer_path else tokenizer_path
        self.TOKENIZER = keras.preprocessing.text.Tokenizer(oov_token='<UNK>', num_words=self.MAX_FEATURES)
        self.GRU_UNITS = 32
        self.MODEL_WEIGHTS_PATH = '../../../../models/model1_sec_nonsec_demo_weights2.h5'
        self.MODEL = None

        print('Loading Tokenizer Vocabulary')
        with open(self.TOKENIZER_PATH, 'rb') as f:
            word2idx = dill.load(f)
        self.TOKENIZER.word_index = word2idx


    def build_model_architecture(self, gru_units=32):
        print('Building Model Architecture')
        self.GRU_UNITS = gru_units if gru_units else self.GRU_UNITS

        input = keras.layers.Input(shape=(self.MAX_LEN, ))
        x = keras.layers.Embedding(self.MAX_FEATURES, self.EMBED_SIZE, trainable=True)(input)
        x = keras.layers.Bidirectional(keras.layers.GRU(self.GRU_UNITS, return_sequences=True,
                                                        reset_after=True, recurrent_activation='sigmoid'))(x)
        x = AttentionLayer(self.MAX_LEN)(x)
        x = keras.layers.Dense(self.GRU_UNITS, activation='relu')(x)
        x = keras.layers.Dropout(rate=0.2)(x)
        output = keras.layers.Dense(1, activation='sigmoid')(x)

        # initialize the model
        model = keras.models.Model(inputs=input, outputs=output)
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        self.MODEL = model


    def load_model_weights(self, model_weights_path=None):
        print('Loading Model Weights')
        self.MODEL_WEIGHTS_PATH = model_weights_path if model_weights_path else self.MODEL_WEIGHTS_PATH
        if not self.MODEL:
            self.build_model_architecture()

        self.MODEL.load_weights(self.MODEL_WEIGHTS_PATH)


    def prepare_inference_data(self, documents):

        doc_sequences = self.TOKENIZER.texts_to_sequences(documents)
        doc_sequences = keras.preprocessing.sequence.pad_sequences(doc_sequences, maxlen=self.MAX_LEN)
        return doc_sequences


    def get_model(self):
        if not self.MODEL:
            raise ModelNotBuiltException("Model doesn't exist. Please build model first")
        else:
            return self.MODEL








