from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class RMSPropOptimizer(optimizer.Optimizer):
    def __init__(self, learning_rate=1e-2, alpha=0.99, epsilon=1e-8, use_locking=False, name="RMSProp"):
        super(RMSPropOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._alpha = alpha
        self._epsilon = epsilon

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._alpha_t = None
        self._epsilon_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._alpha_t = ops.convert_to_tensor(self._alpha, name="alpha")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        m = self.get_slot(var, "m")
        m_t = m.assign(m * alpha_t + (1. - alpha_t) * grad * grad)

        var_update = state_ops.assign_sub(var, lr_t * grad / (tf.sqrt(m_t) + epsilon_t))
        return control_flow_ops.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")