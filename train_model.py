import time
import os
import subprocess
import sys
import re
import argparse
import collections
import gzip
import math
import shutil
import matplotlib.pyplot as plt
import wandb
import numpy as np
from datetime import datetime
import random
from wandb.keras import WandbCallback
import multiprocessing

import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision
import src.metrics as metrics ## switch to src
import src.schedulers
from src.losses import regular_mse
import src.optimizers
import src.schedulers
import src.utils



def tf_tpu_initialize(tpu_name):
    """Initialize TPU and return global batch size for loss calculation
    Args:
        tpu_name
    Returns:
        distributed strategy
    """

    try:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_name)
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)

    except ValueError: # no TPU found, detect GPUs
        strategy = tf.distribute.get_strategy()

    return strategy


"""
having trouble w/ providing organism/step inputs to train/val steps w/o
triggering retracing/metadata resource exhausted errors, so defining
them separately for hg, mm
to do: simplify to two functions w/ organism + mini_batch_step_inputs
"""

def return_train_val_functions(model,
                               optimizer,
                               strategy,
                               precision,
                               orgs,
                               metric_dict,
                               train_steps,
                               val_steps,
                               global_batch_size,
                               gradient_clip):
    """Returns distributed train and validation functions for
    a given list of organisms
    Args:
        orgs: list - either ["hg"] or ["hg", "mm"]
        metric_dict: empty dictionary to populate with organism
                     specific metrics
        train_steps: number of train steps to take in single epoch
        val_steps: number of val steps to take in single epoch
        global_batch_size: # replicas * batch_size_per_replica
    Returns:
        distributed train function
        distributed val function
        metric_dict: dict of tr_loss,val_loss, correlation_stats metrics
                     for input organisms

    return distributed train and val step functions for given organism
    train_steps is the # steps in a single epoch
    val_steps is the # steps to fully iterate over validation set
    """
    #print('parsing_metric_dict')
    ## define the dictionary of metrics
    if precision == 'mixed_bfloat16':
        mixed_precision.set_global_policy('mixed_bfloat16')
    with strategy.scope():

        for org in orgs:
            metric_dict[org + "_tr"] = tf.keras.metrics.Mean(org + "_tr_loss",
                                                             dtype=tf.float32)
            metric_dict[org + "_val"] = tf.keras.metrics.Mean(org + "_val_loss",
                                                              dtype=tf.float32)
            metric_dict[org + "_corr_stats"] = metrics.correlation_stats()

        if orgs == ["hg"]:

            @tf.function
            def dist_train_step(iterator):
                def train_step_hg(inputs):
                    target=inputs['target']
                    model_inputs=inputs['inputs']
                    with tf.GradientTape() as tape:
                        outputs = tf.cast(model(model_inputs,training=True)["hg"],
                                          dtype=tf.float32)
                        loss = tf.reduce_sum(regular_mse(outputs,target),
                                             axis=0) * (1. / global_batch_size)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    #gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip) comment this back in if using adam or adamw
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    metric_dict["hg_tr"].update_state(loss)

                for _ in tf.range(train_steps): ## for loop within @tf.fuction for improved TPU performance
                    strategy.run(train_step_hg, args=(next(iterator),))

            @tf.function
            def dist_val_step(iterator):
                def val_step_hg(inputs):
                    target=inputs['target']
                    tss_tokens = inputs['tss_tokens']
                    model_inputs=inputs['inputs']
                    outputs = tf.cast(model(model_inputs,training=False)["hg"],
                                      dtype=tf.float32)

                    loss = tf.reduce_sum(regular_mse(outputs, target),
                                         axis=0) * (1. / global_batch_size)

                    metric_dict["hg_val"].update_state(loss)

                    return outputs, target, tss_tokens

                ta_pred = tf.TensorArray(tf.float32, size=0, dynamic_size=True) # tensor array to store preds
                ta_true = tf.TensorArray(tf.float32, size=0, dynamic_size=True) # tensor array to store vals
                ta_tss = tf.TensorArray(tf.int32, size=0, dynamic_size=True) # tensor array to store TSS indices
                for _ in tf.range(val_steps): ## for loop within @tf.fuction for improved TPU performance
                    outputs_rep,targets_rep,tss_tokens_rep = strategy.run(val_step_hg,
                                                                            args=(next(iterator),))
                    outputs_reshape = tf.reshape(strategy.gather(outputs_rep, axis=0), [-1]) # reshape to 1D
                    targets_reshape = tf.reshape(strategy.gather(targets_rep, axis=0), [-1])
                    tss_reshape = tf.reshape(strategy.gather(tss_tokens_rep, axis=0), [-1])

                    keep_indices = tf.reshape(tf.where(tf.equal(tss_reshape, 1)), [-1]) # figure out where TSS are
                    targets_sub = tf.gather(targets_reshape, indices=keep_indices)
                    outputs_sub = tf.gather(outputs_reshape, indices=keep_indices)
                    tss_sub = tf.gather(tss_reshape, indices=keep_indices)

                    ta_pred = ta_pred.write(_, outputs_reshape)
                    ta_true = ta_true.write(_, targets_reshape)
                    ta_tss = ta_tss.write(_, tss_reshape)


                preds_all = tf.reshape(ta_pred.gather(np.arange(val_steps)), [-1])
                targets_all = tf.reshape(ta_true.gather(np.arange(val_steps)), [-1])
                tss_all = tf.reshape(ta_tss.gather(np.arange(val_steps)), [-1])

                metric_dict["hg_corr_stats"].update_state(targets_all, preds_all, tss_all) # compute corr stats

            return dist_train_step, dist_val_step, metric_dict


        elif orgs == ["hg", "mm"]:

            @tf.function
            def dist_train_step(hg_iterator, mm_iterator):

                def train_step_hg(inputs):
                    target=inputs['target']
                    model_inputs=inputs['inputs']
                    with tf.GradientTape() as tape:
                        outputs = tf.cast(model(model_inputs,training=True)["hg"],
                                          dtype=tf.float32)
                        loss = tf.reduce_sum(regular_mse(outputs,target),
                                             axis=0) * (1. / global_batch_size)

                    gradients = tape.gradient(loss, model.trainable_variables)
                    #gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip) comment this back in if using adam or adamw
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    metric_dict["hg_tr"].update_state(loss)

                def train_step_mm(inputs):
                    target=inputs['target']
                    model_inputs=inputs['inputs']
                    with tf.GradientTape() as tape:
                        outputs = tf.cast(model(model_inputs,training=True)["mm"],
                                          dtype=tf.float32)
                        loss = tf.reduce_sum(regular_mse(outputs,target),
                                             axis=0) * (1. / global_batch_size)

                    gradients = tape.gradient(loss, model.trainable_variables)
                    #gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip) comment this back in if using adam or adamw
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    metric_dict["mm_tr"].update_state(loss)

                for _ in tf.range(train_steps): ## for loop within @tf.fuction for improved TPU performance
                    strategy.run(train_step_hg, args=(next(hg_iterator),))
                    strategy.run(train_step_mm, args=(next(mm_iterator),))

            @tf.function
            def dist_val_step(hg_iterator, mm_iterator):

                def val_step_hg(inputs):
                    target=inputs['target']
                    tss_tokens = inputs['tss_tokens']
                    model_inputs=inputs['inputs']
                    outputs = tf.cast(model(model_inputs,training=False)["hg"],
                                      dtype=tf.float32)

                    loss = tf.reduce_sum(regular_mse(outputs, target),
                                         axis=0) * (1. / global_batch_size)

                    metric_dict["hg_val"].update_state(loss)
                    return outputs, target, tss_tokens

                def val_step_mm(inputs):
                    target=inputs['target']
                    tss_tokens = inputs['tss_tokens']
                    model_inputs=inputs['inputs']
                    outputs = tf.cast(model(model_inputs,training=False)["mm"],
                                      dtype=tf.float32)

                    loss = tf.reduce_sum(regular_mse(outputs, target),
                                         axis=0) * (1. / global_batch_size)

                    metric_dict["mm_val"].update_state(loss)
                    return outputs, target, tss_tokens

                ta_pred_h = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
                ta_true_h = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
                ta_tss_h = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

                ta_pred_m = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
                ta_true_m = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
                ta_tss_m = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

                for _ in tf.range(val_steps): ## for loop within @tf.fuction for improved TPU performance
                    outputs_rep_h, targets_rep_h,tss_tokens_rep_h = strategy.run(val_step_hg,
                                                                           args=(next(hg_iterator),))
                    outputs_rep_m, targets_rep_m, tss_tokens_rep_m = strategy.run(val_step_mm,
                                                                           args=(next(mm_iterator),))

                    ## all the human tensors
                    outputs_reshape_h = tf.reshape(strategy.gather(outputs_rep_h, axis=0), [-1])
                    targets_reshape_h = tf.reshape(strategy.gather(targets_rep_h, axis=0), [-1])
                    tss_reshape_h = tf.reshape(strategy.gather(tss_tokens_rep_h, axis=0), [-1])

                    keep_indices_h = tf.reshape(tf.where(tf.equal(tss_reshape_h, 1)), [-1])
                    targets_sub_h = tf.gather(targets_reshape_h, indices=keep_indices)
                    outputs_sub_h = tf.gather(outputs_reshape_h, indices=keep_indices)
                    tss_sub_h = tf.gather(tss_reshape_h, indices=keep_indices)

                    ta_pred_h = ta_pred_h.write(_, outputs_sub_h)
                    ta_true_h = ta_true_h.write(_, targets_sub_h)
                    ta_tss_h = ta_tss_h.write(_, tss_sub_h)

                    ## all the mouse tensors
                    outputs_reshape_m = tf.reshape(strategy.gather(outputs_rep_m, axis=0), [-1])
                    targets_reshape_m = tf.reshape(strategy.gather(targets_rep_m, axis=0), [-1])
                    tss_reshape_m = tf.reshape(strategy.gather(tss_tokens_rep_m, axis=0), [-1])

                    keep_indices_m = tf.reshape(tf.where(tf.equal(tss_reshape_m, 1)), [-1])
                    targets_sub_m = tf.gather(targets_reshape_m, indices=keep_indices)
                    outputs_sub_m = tf.gather(outputs_reshape_m, indices=keep_indices)
                    tss_sub_m = tf.gather(tss_reshape_m, indices=keep_indices)

                    ta_pred_m = ta_pred_m.write(_, outputs_sub_m)
                    ta_true_m = ta_true_m.write(_, targets_sub_m)
                    ta_tss_m = ta_tss_m.write(_, tss_sub_m)

                preds_all = tf.reshape(ta_pred_h.gather(np.arange(val_steps)), [-1])
                targets_all = tf.reshape(ta_true_h.gather(np.arange(val_steps)), [-1])
                tss_all = tf.reshape(ta_tss_h.gather(np.arange(val_steps)), [-1])

                preds_all_m = tf.reshape(ta_pred_m.gather(np.arange(val_steps)), [-1])
                targets_all_m = tf.reshape(ta_true_m.gather(np.arange(val_steps)), [-1])
                tss_all_m = tf.reshape(ta_tss_m.gather(np.arange(val_steps)), [-1])


                metric_dict["hg_corr_stats"].update_state(targets_all_h, preds_all_h, tss_all_h)
                metric_dict["mm_corr_stats"].update_state(targets_all_m, preds_all_m, tss_all_m)

            return dist_train_step, dist_val_step, metric_dict
        else:
            raise ValueError('input a proper organism dictionary')



"""
helper functions for returning distributed iter for specific organism
"""

def deserialize(serialized_example, input_length, output_length):
    """
    Deserialize bytes stored in TFRecordFile.
    """
    feature_map = {
        'inputs': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([],tf.string),
        'tss_tokens': tf.io.FixedLenFeature([],tf.string)
    }

    data = tf.io.parse_example(serialized_example, feature_map)

    return {
        'inputs': tf.ensure_shape(tf.io.parse_tensor(data['inputs'],
                                                     out_type=tf.float32),
                                  [input_length,5]),
        'target': tf.ensure_shape(tf.io.parse_tensor(data['target'],
                                                     out_type=tf.float32),
                                  [output_length,]),
        'tss_tokens': tf.ensure_shape(tf.io.parse_tensor(data['tss_tokens'],
                                                     out_type=tf.int32),
                                  [output_length,])
    }


def return_dataset(gcs_path,
                   split,
                   organism,
                   batch,
                   input_length,
                   output_length,
                   num_parallel=8):
    """
    return a tf dataset object for given gcs path
    """
    wc = str(organism) + "*.tfrecords"
    list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                split,
                                                wc)))
    random.shuffle(list_files)
    files = tf.data.Dataset.list_files(list_files)

    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_parallel)

    dataset=dataset.map(lambda record: deserialize(record,input_length,output_length),
                        num_parallel_calls=num_parallel)

    return dataset.repeat().batch(batch).prefetch(tf.data.experimental.AUTOTUNE)


def return_distributed_iterators(heads_dict,
                                 gcs_path,
                                 global_batch_size,
                                 input_length,
                                 output_length,
                                 num_parallel_calls,
                                 strategy):
    """
    returns train + val dictionaries of distributed iterators
    for given heads_dictionary
    """

    data_it_tr_list = []
    data_it_val_list = []

    for org,index in heads_dict.items():
        tr_data = return_dataset(gcs_path,
                                 "train",org,
                                 global_batch_size,
                                 input_length,
                                 output_length,
                                 num_parallel=num_parallel_calls)
        val_data = return_dataset(gcs_path,
                                 "val",org,
                                 global_batch_size,
                                 input_length,
                                 output_length,
                                 num_parallel=num_parallel_calls)

        train_dist = strategy.experimental_distribute_dataset(tr_data)
        val_dist= strategy.experimental_distribute_dataset(val_data)

        tr_data_it = iter(train_dist)
        val_data_it = iter(val_dist)
        data_it_tr_list.append(tr_data_it)
        data_it_val_list.append(val_data_it)
    data_dict_tr = dict(zip(heads_dict.keys(), data_it_tr_list))
    data_dict_val = dict(zip(heads_dict.keys(), data_it_val_list))

    return data_dict_tr, data_dict_val



def early_stopping(current_val_loss,
                   logged_val_losses,
                   current_epoch,
                   save_freq,
                   patience,
                   patience_counter,
                   min_delta,
                   model,
                   save_directory,
                   saved_model_basename):
    """early stopping function
    Args:
        current_val_loss: current epoch val loss
        logged_val_losses: previous epochs val losses
        current_epoch: current epoch number
        save_freq: frequency(in epochs) with which to save checkpoints
        patience: # of epochs to continue w/ stable/increasing val loss
                  before terminating training loop
        patience_counter: # of epochs over which val loss hasn't decreased
        min_delta: minimum decrease in val loss required to reset patience
                   counter
        model: model object
        save_directory: cloud bucket location to save model
        model_parameters: log file of all model parameters
        saved_model_basename: prefix for saved model dir
    Returns:
        stop_criteria: bool indicating whether to exit train loop
        patience_counter: # of epochs over which val loss hasn't decreased
        best_epoch: best epoch so far
    """
    ### check if min_delta satisfied
    previous_loss = logged_val_losses[-1]

    ## if min delta satisfied then log loss
    if previous_loss - current_val_loss > min_delta:
        best_epoch = np.argmin(logged_val_losses)
        ## save current model
        if (current_epoch % save_freq) == 0:
            print('Saving model...')
            model_name = save_directory + "/" + \
                            save_model_base_name + "/" + \
                                iteration + "_" + str(currrent_epoch)
            model.save(model_name)
            ### write to logging file in saved model dir to model parameters and current epoch info

        patience_counter = 0
        stop_criteria = False

    else:
        patience_counter += 1
        if patience_counter >= patience:
            stop_criteria = True

    return stop_criteria, patience_counter, best_epoch


def parse_args(parser):
    """Loads in command line arguments
    """

    parser.add_argument('--tpu_name', dest = 'tpu_name',
                        help='tpu_name')
    parser.add_argument('--tpu_zone', dest = 'tpu_zone',
                        help='tpu_zone')
    parser.add_argument('--wandb_project',
                        dest='wandb_project',
                        help ='wandb_project')
    parser.add_argument('--wandb_user',
                        dest='wandb_user',
                        help ='wandb_user')
    parser.add_argument('--wandb_sweep_name',
                        dest='wandb_sweep_name',
                        help ='wandb_sweep_name')
    parser.add_argument('--gcs_project', dest = 'gcs_project',
                        help='gcs_project')

    # data loading parameters
    parser.add_argument('--gcs_path',
                        dest='gcs_path',
                        help= 'google bucket containing preprocessed data')
    parser.add_argument('--output_heads',
                        dest='output_heads',
                        type=str,
                        help= 'list of organisms(hg,mm,rm)')
    parser.add_argument('--num_parallel', dest = 'num_parallel',
                        type=int, default=multiprocessing.cpu_count(),
                        help='thread count for tensorflow record loading')
    parser.add_argument('--input_length',
                        dest='input_length',
                        type=int,
                        help= 'input_length')
    parser.add_argument('--output_res',
                        dest='output_res',
                        type=int,
                        help= 'output_res')
    parser.add_argument('--output_length',
                        dest='output_length',
                        type=int,
                        help= 'output_length')

    ## training loop parameters
    parser.add_argument('--batch_size', dest = 'batch_size',
                        type=int, help='batch_size')
    parser.add_argument('--num_epochs', dest = 'num_epochs',
                        type=int, help='num_epochs')
    parser.add_argument('--num_warmup_steps', dest = 'num_warmup_steps',
                        type=int, help='num_warmup_steps')
    parser.add_argument('--train_steps', dest = 'train_steps',
                        type=int, help='train_steps')
    parser.add_argument('--val_steps', dest = 'val_steps',
                        type=int, help='val_steps')
    parser.add_argument('--patience', dest = 'patience',
                        type=int, help='patience for early stopping')
    parser.add_argument('--min_delta', dest = 'min_delta',
                        type=float, help='min_delta for early stopping')
    parser.add_argument('--model_save_dir',
                        dest='model_save_dir',
                        type=str)
    parser.add_argument('--model_save_basename',
                        dest='model_save_basename',
                        type=str)
## parameters to sweep over
    parser.add_argument('--lr_schedule',
                        dest = 'lr_schedule',
                        type=str)
    parser.add_argument('--lr_base',
                        dest='lr_base',
                        help='lr_base')
    parser.add_argument('--warmup_lr',
                        dest='warmup_lr',
                        help= 'warmup_lr')
    parser.add_argument('--optimizer',
                        dest='optimizer',
                        help= 'optimizer, one of adafactor, adam, or adamW')
    parser.add_argument('--gradient_clip',
                        dest='gradient_clip',
                        type=str,
                        help= 'gradient_clip')
    parser.add_argument('--precision',
                        dest='precision',
                        type=str,
                        help= 'bfloat16 or float32') ### need to implement this actually
    parser.add_argument('--weight_decay',
                        dest='weight_decay',
                        type=str,
                        help= 'weight_decay')


    # network hyperparameters
    parser.add_argument('--conv_channel_list',
                        dest='conv_channel_list',
                        help= 'conv_channel_list')
    parser.add_argument('--dropout',
                        dest='dropout',
                        help= 'dropout')
    parser.add_argument('--num_transformer_layers',
                        dest='num_transformer_layers',
                        help= 'num_transformer_layers')
    parser.add_argument('--num_heads',
                        dest='num_heads',
                        help= 'num_heads')
    parser.add_argument('--momentum',
                        dest='momentum',
                        type=str,
                        help= 'batch norm momentum')
    parser.add_argument('--num_random_features',
                        dest='num_random_features',
                        type=str,
                        help= 'num_random_features')
    parser.add_argument('--kernel_transformation',
                        dest='kernel_transformation',
                        help= 'kernel_transformation')
    parser.add_argument('--hidden_size',
                        dest='hidden_size',
                        type=str,
                        help= 'hidden size for transformer' + \
                                'should be equal to last conv layer filters')
    parser.add_argument('--conv_filter_size',
                        dest='conv_filter_size',
                        help= 'conv_filter_size')



    args = parser.parse_args()
    return parser
