"""Functions and classes related to LAMB optimizer (weight updates)."""

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
import re


class LAMBOptimizer(Optimizer):
    """LAMBOptimizer optimizer.
        Default parameters follow those provided in the original paper.
        # Arguments
            lr: float >= 0. Learning rate.
            beta_1: float, 0 < beta < 1. Generally close to 1.
            beta_2: float, 0 < beta < 1. Generally close to 1.
            epsilon: float >= 0. Fuzz factor. If `None`, defaults to 1e-6.
            weight_decay: float >= 0. Weight decay regularization.
            exclude_from_weight_decay: Layers to exclude from weight decay while pre-training.
        # References
            - [Reducing BERT Pre-Training Time from 3 Days to 76 Minutes]
              (https://arxiv.org/abs/1904.00962)
            - We use v4 of the paper to implement this (which is latest as of now (Nov 26th 2019))
            - Code is different across paper revisions, so please make sure to refer to v4
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-6, weight_decay=0., exclude_from_weight_decay=None,
                 **kwargs):
        super(LAMBOptimizer, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, dtype='float32', name='lr')
            self.beta_1 = K.variable(beta_1, dtype='float32', name='beta_1')
            self.beta_2 = K.variable(beta_2, dtype='float32', name='beta_2')
            self.epsilon = K.variable(epsilon, dtype='float32', name='epsilon')
            self.weight_decay = K.variable(weight_decay, dtype='float32', name='weight_decay')

        self.exclude_from_weight_decay = exclude_from_weight_decay

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

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

            if self._do_use_weight_decay(p.name):
                wd = self.weight_decay * p
                p_dash = p_dash + wd

            r1 = linalg_ops.norm(p, ord=2)
            r2 = linalg_ops.norm(p_dash, ord=2)

            r = array_ops.where(math_ops.greater(r1, 0), array_ops.where(
                math_ops.greater(r2, 0), (r1 / r2), 1.0), 1.0)
            
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

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.weight_decay is None:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

