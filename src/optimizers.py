"""Orginally implementation from https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/adafactor.py """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow_addons.utils import keras_utils
import sys


@tf.keras.utils.register_keras_serializable()
class AdafactorOptimizer(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the Adafactor algorithm.
    Adafactor is described in https://arxiv.org/abs/1804.04235.

    Several parts of this algorithm are configurable from the initializer.
        multiply_by_parameter_scale:    If True, then compute absolute_update_scale
            as described above.  If False, let absolute_update_scale be the externally
            supplied learning_rate.
        learning_rate: represents relative_update_scale if
            multiply_by_parameter_scale==True, or absolute_update_scale if
            multiply_by_parameter_scale==False.
        decay_rate: Decay rate of the second moment estimator (varies by step_num).
            This should be set to a function such that:
            1-1/(step_num + 1) <= decay_rate(step_num) < 1.0
        beta1: enables momentum, as in Adam.    Uses extra memory if nonzero.
        clipping_threshold: should be >=1.0 or None for no update clipping
        factored: whether to factor the second-moment estimator.    True means
            less memory usage.
    """
    def __init__(self,
                multiply_by_parameter_scale=False,
                learning_rate=None,
                decay_rate=None,
                beta1=0.0,
                clipping_threshold=1.0,
                factored=True,
                use_locking=False,
                name="Adafactor",
                epsilon1=1e-30,
                epsilon2=1e-3,
                **kwargs):
        """Construct a new Adafactor optimizer.
        See class comment.
        Args:
            multiply_by_parameter_scale: a boolean
            learning_rate: an optional Scalar.
            decay_rate: an optional Scalar.
            beta1: a float value between 0 and 1
            clipping_threshold: an optional float >= 1
            factored: a boolean - whether to use factored second-moment estimator
                for 2d variables
            use_locking: If True use locks for update operations.
            name: Optional name for the operations created when applying gradients.
                Defaults to "AdafactorOptimizer".
            epsilon1: Regularization constant for squared gradient.
            epsilon2: Regularization constant for parameter scale.
        Raises:
            ValueError: if absolute_update_scale and relative_update_scale_fn are both
                present or both absent.
        """
        super(AdafactorOptimizer, self).__init__(name=name, **kwargs)

        # Set Flags
        self.multiply_by_parameter_scale = multiply_by_parameter_scale
        self.factored = factored
        self.use_locking = use_locking
        self.has_beta_1 = (beta1!=0.0)

        # Set defaults
        if learning_rate is None:
            learning_rate = self._learning_rate_default(multiply_by_parameter_scale)

        if decay_rate is None:
            decay_rate = self._decay_rate_default()

        # Set Hypers
        self._set_hyper("decay_rate",decay_rate)
        self._set_hyper("learning_rate",learning_rate)
        self._set_hyper("beta1", beta1)
        self._set_hyper("clipping_threshold",clipping_threshold)
        self._set_hyper("factored",factored)
        self._set_hyper("epsilon1",epsilon1)
        self._set_hyper("epsilon2",epsilon2)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        learning_rate_t = tf.identity(self._get_hyper("learning_rate", var_dtype))
        decay_rate_t = tf.identity(self._get_hyper("decay_rate", var_dtype))
        beta_1_t = tf.identity(self._get_hyper("beta1", var_dtype))
        clipping_threshold_t = tf.identity(self._get_hyper("clipping_threshold", var_dtype))
        epsilon1_t = tf.identity(self._get_hyper("epsilon1", var_dtype))
        epsilon2_t = tf.identity(self._get_hyper("epsilon2", var_dtype))

        apply_state[(var_device, var_dtype)].update(
            dict(
                learning_rate = learning_rate_t,
                decay_rate = decay_rate_t,
                beta1 = beta_1_t,
                clipping_threshold = clipping_threshold_t,
                epsilon1 = epsilon1_t,
                epsilon2 = epsilon2_t,
            )
        )


    def get_config(self):
        config = {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "decay_rate": self._serialize_hyperparameter("decay_rate"),
                "beta1": self._serialize_hyperparameter("beta1"),
                "clipping_threshold": self._serialize_hyperparameter("clipping_threshold"),
                "epsilon1": self._serialize_hyperparameter("epsilon1"),
                "epsilon2": self._serialize_hyperparameter("epsilon2")
        }
        base_config = super(AdafactorOptimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _should_use_factored_second_moment_estimate(self, shape):
        """Should we use a factored second moment estimator.
        Based on the shape of the variable.
        Args:
            shape: a list of integers
        Returns:
            a boolean
        """
        return self.factored and len(shape) >= 2


    def _create_slots(self, var_list):
        for var in var_list:
            shape = var.get_shape().as_list()
            if self.has_beta_1:
                self.add_slot(var, "m")
            if self._should_use_factored_second_moment_estimate(shape):
                r_val = tf.zeros(shape[:-1], dtype=tf.float32)
                c_val = tf.zeros(shape[:-2] + shape[-1:], dtype=tf.float32)
                self.add_slot(var, "vr", initializer=r_val)
                self.add_slot(var, "vc", initializer=c_val)
            else:
                v_val = tf.zeros(shape, dtype=tf.float32)
                self.add_slot(var, "v", initializer=v_val)

    def _apply_dense(self, grad, var):
        return self._resource_apply_dense(grad, var)

    def _apply_sparse(self, grad, var):
        return self._apply_dense(tf.convert_to_tensor(grad), var)

    def _resource_apply_sparse(self, grad, handle, indices):
        return self._resource_apply_dense(
                tf.convert_to_tensor(tf.IndexedSlices(grad, indices, tf.shape(handle))),
                handle)

    def _parameter_scale(self, var):
        """Estimate the scale of the parameters from the current values.
        We include a minimum value of 0.001 to give it a chance to escape 0
        if it was zero-initialized.
        Instead of using the value, we could impute the scale from the shape,
        as initializers do.
        Args:
            var: a variable or Tensor.
        Returns:
            a Scalar
        """
        tf.cast(var,tf.float32)
        testy = tf.maximum(reduce_rms(var), self._get_hyper("epsilon2"))
        tf.cast(testy,tf.float32)
        return tf.maximum(reduce_rms(var), self._get_hyper("epsilon2"))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        grad = tf.cast(grad,tf.float32)
        grad_squared = tf.square(grad) + coefficients["epsilon1"]
        grad_squared_mean = tf.reduce_mean(grad_squared)
        decay_rate = coefficients["decay_rate"]
        update_scale = coefficients["learning_rate"]
        old_val = var
        if self.multiply_by_parameter_scale:
            scale_factor = self._parameter_scale(old_val)
            update_scale *= tf.cast(scale_factor, tf.float32)
        # HACK: Make things dependent on grad.
        # This confounds the XLA rewriter and keeps it from fusing computations
        # across different variables.  This fusion is a bad for HBM usage, since
        # it causes the gradients to persist in memory.
        decay_rate += grad_squared_mean * 1e-30
        update_scale += grad_squared_mean * 1e-30
        # END HACK
        mixing_rate = 1.0 - decay_rate
        shape = var.get_shape().as_list()
        updates = []
        if self._should_use_factored_second_moment_estimate(shape):
            grad_squared_row_mean = tf.reduce_mean(grad_squared, -1)
            vr = self.get_slot(var, "vr")
            new_vr = (decay_rate * vr + mixing_rate * grad_squared_row_mean)
            vr_update = vr.assign(new_vr, use_locking = self.use_locking)
            updates.append(vr_update)

            grad_squared_col_mean = tf.reduce_mean(grad_squared, -2)
            vc = self.get_slot(var, "vc")
            new_vc = (decay_rate * vc + mixing_rate * grad_squared_col_mean)
            vc_update = vc.assign(new_vc, use_locking=self.use_locking)
            updates.append(vc_update)

            long_term_mean = tf.reduce_mean(new_vr, -1, keepdims=True)
            r_factor = tf.math.rsqrt(new_vr / long_term_mean)
            c_factor = tf.math.rsqrt(new_vc)
            x = grad * tf.expand_dims(r_factor, -1) * tf.expand_dims(c_factor, -2)
        else:
            v = self.get_slot(var, "v")
            new_v = decay_rate * v + mixing_rate * grad_squared
            v_update = v.assign(new_v, use_locking = self.use_locking)
            updates = [v_update]
            x = grad * tf.math.rsqrt(new_v)

        if coefficients["clipping_threshold"] is not None:
            clipping_denom = tf.maximum(1.0, reduce_rms(x) / coefficients["clipping_threshold"])
            x /= clipping_denom
        subtrahend = update_scale * x

        if self.has_beta_1:
            m = self.get_slot(var, "m")
            new_m = coefficients["beta1"] * tf.to_float(m) + (1.0 - coefficients["beta1"]) * subtrahend
            subtrahend = new_m
            new_m = self._cast_like(new_m, var)
            m_update_value = m.assign(new_m, use_locking=self.use_locking)
            updates.append(m_update_value)

        new_val = tf.cast(old_val,tf.float32) - subtrahend
        new_val = var.assign(new_val, use_locking=self.use_locking)
        updates = [new_val] + updates
        return tf.group(*updates)

    def _cast_like(self,x, y):
        """Cast x to y's dtype, if necessary. Grabbed from tensor2tensor/layers/common_layers"""
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)

        if x.dtype.base_dtype == y.dtype.base_dtype:
            return x

        cast_x = tf.cast(x, y.dtype)
        if cast_x.device != x.device:
            x_name = "(eager Tensor)"
            try:
                x_name = x.name
            except AttributeError:
                pass
            tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x_name,
                    x.device, cast_x.device)
            return cast_x

    def _decay_rate_default(self):
        return self._adafactor_decay_rate_pow(0.8)

    def _learning_rate_default(self, multiply_by_parameter_scale):
        learning_rate = tf.minimum(tf.math.rsqrt(self.step_num() + 1.0), 0.01)
        if not multiply_by_parameter_scale:
            learning_rate *= 0.05
        return learning_rate


    def _adafactor_decay_rate_adam(self, beta2):
        """Second-moment decay rate like Adam, subsuming the correction factor.
        Args:
            beta2: a float between 0 and 1
        Returns:
            a scalar
        """
        t = tf.cast(self.iterations,tf.float32) + 1.0
        decay = beta2 * (1.0 - tf.pow(beta2, t - 1.0)) / (1.0 - tf.pow(beta2, t))
        # decay = tf.cond(tf.equal(t, 1.0), lambda: beta2, lambda: decay)
        return decay


    def _adafactor_decay_rate_pow(self, exponent):
        """Second moment decay rate where memory-length grows as step_num^exponent.
        Args:
            exponent: a float between 0 and 1
        Returns:
            a scalar
        """
        return 1.0 - tf.pow((self.step_num() + 1.0), -exponent)


    def step_num(self):
        return tf.cast(self.iterations,tf.float32)


def adafactor_optimizer_from_hparams(hparams, lr):
    """Create an Adafactor optimizer based on model hparams.
    Args:
        hparams: model hyperparameters
        lr: learning rate scalar.
    Returns:
        an AdafactorOptimizer
    Raises:
        ValueError: on illegal values
    """
    if hparams.optimizer_adafactor_decay_type == "adam":
        decay_rate = self._adafactor_decay_rate_adam(
                hparams.optimizer_adafactor_beta2)
    elif hparams.optimizer_adafactor_decay_type == "pow":
        decay_rate = adafactor_decay_rate_pow(
                hparams.optimizer_adafactor_memory_exponent)
    else:
        raise ValueError("unknown optimizer_adafactor_decay_type")
    if hparams.weight_dtype == "bfloat16":
        parameter_encoding = quantization.EighthPowerEncoding()
    else:
        parameter_encoding = None
    return AdafactorOptimizer(
            multiply_by_parameter_scale=(
                hparams.optimizer_adafactor_multiply_by_parameter_scale),
            learning_rate=lr,
            decay_rate=decay_rate,
            beta1=hparams.optimizer_adafactor_beta1,
            clipping_threshold=hparams.optimizer_adafactor_clipping_threshold,
            factored=hparams.optimizer_adafactor_factored,
            simulated_quantize_bits=getattr(
                hparams, "simulated_parameter_quantize_bits", 0),
            parameter_encoding=parameter_encoding,
            use_locking=False,
            name="Adafactor")

def reduce_rms(x):
    return tf.math.sqrt(tf.reduce_mean(tf.square(x)))