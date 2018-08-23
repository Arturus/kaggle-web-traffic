# Copyright 2017 Francesco Orabona. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""
COntinuos COin Betting (COCOB) optimizer
See 'Training Deep Networks without Learning Rates Through Coin Betting'
https://arxiv.org/abs/1705.07795 
"""

from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.training.optimizer import Optimizer
import tensorflow as tf



class COCOB(Optimizer):
    def __init__(self, alpha=100, use_locking=False, name='COCOB'):
        '''
        constructs a new COCOB optimizer
        '''
        super(COCOB, self).__init__(use_locking, name)
        self._alpha = alpha

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                gradients_sum = constant_op.constant(0,
                                                     shape=v.get_shape(),
                                                     dtype=v.dtype.base_dtype)
                grad_norm_sum = constant_op.constant(0,
                                                     shape=v.get_shape(),
                                                     dtype=v.dtype.base_dtype)
                L = constant_op.constant(1e-8, shape=v.get_shape(), dtype=v.dtype.base_dtype)
                tilde_w = constant_op.constant(0.0, shape=v.get_shape(), dtype=v.dtype.base_dtype)
                reward = constant_op.constant(0.0, shape=v.get_shape(), dtype=v.dtype.base_dtype)

            self._get_or_make_slot(v, L, "L", self._name)
            self._get_or_make_slot(v, grad_norm_sum, "grad_norm_sum", self._name)
            self._get_or_make_slot(v, gradients_sum, "gradients_sum", self._name)
            self._get_or_make_slot(v, tilde_w, "tilde_w", self._name)
            self._get_or_make_slot(v, reward, "reward", self._name)

    def _apply_dense(self, grad, var):
        gradients_sum = self.get_slot(var, "gradients_sum")
        grad_norm_sum = self.get_slot(var, "grad_norm_sum")
        tilde_w = self.get_slot(var, "tilde_w")
        L = self.get_slot(var, "L")
        reward = self.get_slot(var, "reward")

        L_update = tf.maximum(L, tf.abs(grad))
        gradients_sum_update = gradients_sum + grad
        grad_norm_sum_update = grad_norm_sum + tf.abs(grad)
        reward_update = tf.maximum(reward - grad * tilde_w, 0)
        new_w = -gradients_sum_update / (
        L_update * (tf.maximum(grad_norm_sum_update + L_update, self._alpha * L_update))) * (reward_update + L_update)
        var_update = var - tilde_w + new_w
        tilde_w_update = new_w

        gradients_sum_update_op = state_ops.assign(gradients_sum, gradients_sum_update)
        grad_norm_sum_update_op = state_ops.assign(grad_norm_sum, grad_norm_sum_update)
        var_update_op = state_ops.assign(var, var_update)
        tilde_w_update_op = state_ops.assign(tilde_w, tilde_w_update)
        L_update_op = state_ops.assign(L, L_update)
        reward_update_op = state_ops.assign(reward, reward_update)

        return control_flow_ops.group(*[gradients_sum_update_op,
                                        var_update_op,
                                        grad_norm_sum_update_op,
                                        tilde_w_update_op,
                                        reward_update_op,
                                        L_update_op])

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)
