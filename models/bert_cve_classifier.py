"""Fine-tuned BERT Classification Model for predicting
   probable vulnerabilities (CVEs)"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import tensorflow_hub as tf_hub
from tensorflow.keras import backend as K
from .optimizers import LAMBOptimizer

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

    def call(self, inputs, *args, **kwargs):
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
                      optimizer=LAMBOptimizer(exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]), 
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