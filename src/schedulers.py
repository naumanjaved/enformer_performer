import inspect
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable
import numpy as np
import tensorflow.experimental.numpy as tnp
import tensorflow as tf
#tnp.experimental_enable_numpy_behavior(prefer_float32=True)

from tensorflow.keras import layers as kl

@tf.keras.utils.register_keras_serializable()
class cosine_decay_w_warmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay schedule with warm up period.
    mod from https://github.com/Tony607/Keras_Bag_of_Tricks/blob/master/warmup_cosine_decay_scheduler.py
    """
    def __init__(self,
                 peak_lr,
                 initial_learning_rate,
                 total_steps,
                 warmup_steps,
                 hold_peak_rate_steps,
                 name = None):
        self.peak_lr = peak_lr
        self.initial_learning_rate = initial_learning_rate
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.hold_peak_rate_steps = hold_peak_rate_steps
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "cosine_decay_w_warmup"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name = "initial_learning_rate")
        dtype = initial_learning_rate.dtype
        peak_lr = tf.cast(self.peak_lr, dtype)
        total_steps = tf.cast(self.total_steps,dtype)
        warmup_steps = tf.cast(self.warmup_steps,dtype)
        hold_peak_rate_steps = tf.cast(self.hold_peak_rate_steps,dtype)
        
        global_step_recomp = tf.cast(step, dtype)
        
        learning_rate = 0.5 * peak_lr * (1. + tnp.cos(
            tnp.pi * (global_step_recomp - warmup_steps - hold_peak_rate_steps
                    ) / (total_steps - warmup_steps - hold_peak_rate_steps)))
                                         
        learning_rate = tf.cond(tf.math.greater(hold_peak_rate_steps,0),
                                lambda: tnp.where(global_step_recomp > warmup_steps + hold_peak_rate_steps,
                                     learning_rate, peak_lr),
                                lambda: learning_rate)
                                
        slope = (initial_learning_rate - peak_lr) / warmup_steps
        warmup_rate = slope * global_step_recomp + initial_learning_rate
        learning_rate = tf.cond(tf.math.greater(warmup_steps, 0),
                                lambda: tnp.where(global_step_recomp < warmup_steps, warmup_rate,
                                         learning_rate),
                                lambda: learning_rate)
        return tnp.where(global_step_recomp > total_steps, initial_learning_rate, learning_rate)
                                         
    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "initial_learning_rate": self.initial_learning_rate,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "hold_peak_rate_steps": self.hold_peak_rate_steps,
            "name": self.name
        }

'''
cosine decay w/ warmup 
taken from
https://github.com/Tony607/Keras_Bag_of_Tricks/blob/master/warmup_cosine_decay_scheduler.py
'''
#@tf.keras.utils.register_keras_serializable
def cosine_decay_with_warmup_func(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)