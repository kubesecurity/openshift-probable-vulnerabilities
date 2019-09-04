"""Fine-tuned BERT Classification Model for predicting
   probable vulnerabilities (CVEs)"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import tensorflow_hub as tf_hub
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Optimizer


class LAMBOptimizer(Optimizer):
    """LAMBOptimizer optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to 1e-6.
        weight_decay: float >= 0. Weight decay regularization.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Reducing BERT Pre-Training Time from 3 Days to 76 Minutes]
          (https://arxiv.org/abs/1904.00962)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, weight_decay=0.01, decay=0., **kwargs):
        super(LAMBOptimizer, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = 1e-6
        self.epsilon = epsilon
        self.initial_decay = decay
        self.weight_decay = weight_decay

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            m_t_hat = m_t / (1. - K.pow(self.beta_1, t))
            v_t_hat = v_t / (1. - K.pow(self.beta_2, t))

            p_dash = m_t_hat / (K.sqrt(v_t_hat + self.epsilon))

            if self.weight_decay > 0.:
                wd = self.weight_decay * p
                p_dash = p_dash + wd

            r1 = K.sqrt(K.sum(K.square(p)))
            r2 = K.sqrt(K.sum(K.square(p_dash)))

            r = tf.where(tf.greater(r1, 0.),
                         tf.where(tf.greater(r2, 0.),
                                  r1 / r2,
                                  1.0),
                         1.0)
            # r = r1 / r2
            eta = r * lr

            p_t = p - eta * p_dash

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'weight_decay': self.weight_decay}
        base_config = super(LAMBOptimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BertLayer(tf.keras.layers.Layer):
    
    def __init__(self, bert_model_path, n_fine_tune_encoders=10, **kwargs,):
        
        self.n_fine_tune_encoders = n_fine_tune_encoders
        self.trainable = True
        # change only based on base bert output layer shape
        self.output_size = 768
        self.bert_path = bert_model_path
        super(BertLayer, self).__init__(**kwargs)

        
    def build(self, input_shape):
        print('Loading Base BERT Model')
        self.bert = tf_hub.Module(self.bert_path,
                                  trainable=self.trainable, 
                                  name=f"{self.name}_module")

        # Remove unused layers
        # CLS layers cause an error if you try to tune them
        trainable_vars = self.bert.variables
        trainable_vars = [var for var in trainable_vars 
                                  if not "/cls/" in var.name]
        trainable_layers = ["embeddings", "pooler/dense"]


        # Select how many layers to fine tune
        # we fine-tune all layers per encoder
        # by default we tune all 10 encoders
        for i in range(self.n_fine_tune_encoders+1):
            trainable_layers.append(f"encoder/layer_{str(10 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [var for var in trainable_vars
                                  if any([l in var.name 
                                              for l in trainable_layers])]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:# and 'encoder/layer' not in var.name:
                self._non_trainable_weights.append(var)
        print('Trainable layers:', len(self._trainable_weights))
        print('Non Trainable layers:', len(self._non_trainable_weights))

        super(BertLayer, self).build(input_shape)

        
    def call(self, inputs):
        print('Constructing Base BERT architecture')
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(input_ids=input_ids, 
                           input_mask=input_mask, 
                           segment_ids=segment_ids)
        
        pooled = self.bert(inputs=bert_inputs, 
                           signature="tokens", 
                           as_dict=True)["pooled_output"]

        return pooled

    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size) 


class ModelNotBuiltException(Exception):
    pass


class BERTClassifier:
    
    def __init__(self, bert_model_path, max_seq_length=128, 
                 n_fine_tune_encoders=10, model_weights_path=None):
        self.bert_path = bert_model_path
        self.max_seq_length = max_seq_length
        self.n_fine_tune_encoders = n_fine_tune_encoders
        self.model_estimator = None
        self.model_weights_path = model_weights_path
    
    def build_model_architecture(self): 
        print('Build BERT Classifier CVE Model Architecture')
        inp_id = tf.keras.layers.Input(shape=(self.max_seq_length,), 
                                       name="input_ids")
        inp_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), 
                                         name="input_masks")
        inp_segment = tf.keras.layers.Input(shape=(self.max_seq_length,), 
                                            name="segment_ids")
        bert_inputs = [inp_id, inp_mask, inp_segment]

        bert_output = BertLayer(bert_model_path=self.bert_path, 
                                n_fine_tune_encoders=self.n_fine_tune_encoders)(bert_inputs)

        dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
        pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

        model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
        model.compile(loss='binary_crossentropy', 
                      optimizer=LAMBOptimizer(lr=5e-4, weight_decay=5e-5), 
                      metrics=['accuracy'])    
        self.model_estimator = model
        
    
    def load_model_weights(self, model_weights_path=None):
        print('Loading BERT Classifier CVE Model Weights')
        self.model_weights_path = model_weights_path or self.model_weights_path
        if not self.model_estimator:
            self.build_model_architecture()
        self.model_estimator.load_weights(self.model_weights_path)
            
    
    def get_model(self):
        if not self.model_estimator:
            raise ModelNotBuiltException(
                "BERT Classifier CVE Model doesn't exist. Please build model first")
        else:
            return self.model_estimator