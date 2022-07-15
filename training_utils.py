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

import multiprocessing

import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision
import src.metrics as metrics ## switch to src 
import src.schedulers
from src.losses import regular_mse,abs_mse,poisson
import src.optimizers
import src.schedulers
import pandas as pd
import src.utils
import seaborn as sns

from scipy.stats.stats import pearsonr
from scipy.stats import linregress
from scipy import stats
import keras.backend as kb

import scipy.special
import scipy.stats
import scipy.ndimage

tf.keras.backend.set_floatx('float32')

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
consolidate into single simpler function
"""


def return_train_val_functions_hg(model,
                                   optimizer,
                                   strategy,
                                   metric_dict,
                                   train_steps, 
                                   val_steps, 
                                   global_batch_size,
                                   gradient_clip):
                                   #use_prior):
    """Returns distributed train and validation functions for
    a given list of organisms
    Args:
        model: model object
        optimizer: optimizer object
        metric_dict: empty dictionary to populate with organism
                     specific metrics
        train_steps: number of train steps to take in single epoch
        val_steps: number of val steps to take in single epoch
        global_batch_size: # replicas * batch_size_per_replica
        gradient_clip: gradient clip value to be applied in case of adam/adamw optimizer
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
    metric_dict["hg_tr"] = tf.keras.metrics.Mean("hg_tr_loss",
                                                 dtype=tf.float32)
    metric_dict["hg_val"] = tf.keras.metrics.Mean("hg_val_loss",
                                                  dtype=tf.float32)
    metric_dict["hg_corr_stats"] = metrics.correlation_stats_gene_centered(name='hg_corr_stats')
    

    
    def dist_train_step(iterator):
        @tf.function(jit_compile=True)
        def train_step_hg(inputs):
            target=tf.cast(inputs['target'],dtype=tf.float32)
            #tf.print(target, output_stream=sys.stderr)
            seq_inputs=tf.cast(inputs['inputs'],
                                 dtype=tf.bfloat16)
            atac_inputs =tf.cast(inputs['atac'],
                                 dtype=tf.bfloat16)
            tf_input = tf.cast(inputs['TF_acc'],
                               dtype=tf.bfloat16)
            input_tuple = seq_inputs,atac_inputs,tf_input
            
            with tf.GradientTape() as tape:
                with tf.GradientTape() as input_grad_tape:
                    input_grad_tape.watch(input_tuple)
                    #input_grad_tape.watch(atac_inputs)
                    #input_grad_tape.watch(tf_input)
                    output = model(input_tuple,training=True)[0]["hg"]
                    output = tf.cast(output,dtype=tf.float32)
                loss = tf.reduce_sum(regular_mse(output, target),
                                     axis=0) * (1. / global_batch_size)
                """
                print(loss)
                
                # Compute gradients of the output with respect to the input
                if use_prior:
                    input_grads = input_grad_tape.gradient(output,input_tuple)
                    input_grads = input_grads[0] * seq_inputs # Gradient * input
                    #input_grads = tf.cast(input_grads,dtype=tf.float32)
                    fourier_loss = fourier_att_prior_loss(output, input_grads)

                    print(fourier_loss)
                    loss = loss + fourier_loss
                    print(loss)
                """
                
            #print(model.trainable_variables)
            gradients = tape.gradient(loss, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip) #comment this back in if using adam or adamW
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            metric_dict["hg_tr"].update_state(loss)

        for _ in tf.range(train_steps): ## for loop within @tf.fuction for improved TPU performance
            strategy.run(train_step_hg, args=(next(iterator),))

    
    def dist_val_step(iterator):
        @tf.function(jit_compile=True)
        def val_step_hg(inputs):
            target=tf.cast(inputs['target'],dtype=tf.float32)
            seq_inputs=tf.cast(inputs['inputs'],
                                 dtype=tf.float32)
            atac_inputs =tf.cast(inputs['atac'],
                                 dtype=tf.float32)
            tf_input = tf.cast(inputs['TF_acc'],
                               dtype=tf.float32)
            input_tuple = seq_inputs,atac_inputs,tf_input
            
            cell_type = inputs['cell_type']
            gene_map = inputs['gene_encoded']

            output = tf.cast(model(input_tuple,training=False)[0]["hg"],
                              dtype=tf.float32)

            loss = tf.reduce_sum(regular_mse(output, target),
                                 axis=0) * (1. / global_batch_size)

            metric_dict["hg_val"].update_state(loss)

            return target, output, cell_type, gene_map
    
        ta_pred = tf.TensorArray(tf.float32, size=0, dynamic_size=True) # tensor array to store preds
        ta_true = tf.TensorArray(tf.float32, size=0, dynamic_size=True) # tensor array to store vals
        ta_celltype = tf.TensorArray(tf.int32, size=0, dynamic_size=True) # tensor array to store preds
        ta_genemap = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        
        for _ in tf.range(val_steps): ## for loop within @tf.fuction for improved TPU performance
            target_rep, output_rep, cell_type_rep, gene_map_rep = strategy.run(val_step_hg,
                                                                               args=(next(iterator),))
            
            target_reshape = tf.reshape(strategy.gather(target_rep, axis=0), [-1]) # reshape to 1D
            output_reshape = tf.reshape(strategy.gather(output_rep, axis=0), [-1])
            cell_type_reshape = tf.reshape(strategy.gather(cell_type_rep, axis=0), [-1])
            gene_map_reshape = tf.reshape(strategy.gather(gene_map_rep, axis=0), [-1])

            ta_pred = ta_pred.write(_, output_reshape)
            ta_true = ta_true.write(_, target_reshape)
            ta_celltype = ta_celltype.write(_, cell_type_reshape)
            ta_genemap = ta_genemap.write(_, gene_map_reshape)

        metric_dict["hg_corr_stats"].update_state(ta_true.concat(),
                                                  ta_pred.concat(),
                                                  ta_celltype.concat(),
                                                  ta_genemap.concat())
        ta_pred.close()
        ta_true.close()
        ta_celltype.close()
        ta_genemap.close()

    return dist_train_step, dist_val_step, metric_dict


def deserialize(serialized_example, input_length, 
                num_TFs,max_shift,output_type):
    """
    Deserialize bytes stored in TFRecordFile.
    """
    feature_map = {
        'atac': tf.io.FixedLenFeature([], tf.string),
        'exons': tf.io.FixedLenFeature([],tf.string),
        'sequence': tf.io.FixedLenFeature([],tf.string),
        'TPM': tf.io.FixedLenFeature([],tf.string),
        'TPM_uqn': tf.io.FixedLenFeature([],tf.string),
        'gene_mean': tf.io.FixedLenFeature([],tf.string),
        'gene_std': tf.io.FixedLenFeature([],tf.string),
        'TF_acc': tf.io.FixedLenFeature([], tf.string)
    }
    
    data = tf.io.parse_example(serialized_example, feature_map)

    ### stochastic sequence shift and gaussian noise
    shift = random.randrange(0,max_shift,1)
    input_seq_length = input_length + max_shift
    interval_end = input_length + shift
    
    ### rev_comp
    rev_comp = random.randrange(0,2)
    
    exons = tf.ensure_shape(tf.io.parse_tensor(data['exons'],
                                              out_type=tf.int32),
                            [input_seq_length,])
    exons = tf.cast(tf.slice(exons, [shift],[input_length]),dtype=tf.float32)
    
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                              out_type=tf.float32),
                           [input_seq_length,])
    atac = tf.slice(atac, [shift],[input_length])
    atac = tf.nn.dropout(atac, rate=0.05)*(0.95)
    atac = atac + tf.math.abs(tf.random.normal(exons.shape, 
                                   mean=0.0, 
                                   stddev=5.0e-01, dtype=tf.float32))
    atac=tf.expand_dims(atac,1)
    sequence = one_hot(tf.strings.substr(data['sequence'],
                                 shift,input_length))
    
    if rev_comp == 1:
        atac = tf.reverse(atac,[0])
        exons = tf.reverse(exons,[0])
        sequence = rev_comp_one_hot(tf.strings.substr(data['sequence'],
                                                      shift,input_length))
    
    sequence = tf.nn.dropout(sequence, rate=0.05)*(0.95)
    TF_acc = tf.ensure_shape(tf.io.parse_tensor(data['TF_acc'],
                                              out_type=tf.float32),
                             [num_TFs,])
    TF_acc = tf.nn.dropout(TF_acc, rate=0.01)*(0.99)
    
    #inputs = tf.concat([tf.expand_dims(atac, 1), sequence], axis=1)
    inputs = tf.concat([tf.expand_dims(exons,1), sequence], axis=1)

    TPM = tf.io.parse_tensor(data['TPM'],out_type=tf.float32)
    TPM_uqn = tf.io.parse_tensor(data['TPM_uqn'],out_type=tf.float32)
    gene_mean = tf.io.parse_tensor(data['gene_mean'],out_type=tf.float32)
    gene_std = tf.io.parse_tensor(data['gene_std'],out_type=tf.float32)

    if output_type == 'logTPM':
        target = log2(1.0 + tf.math.maximum(0.0,TPM))
    elif output_type == 'zTPM':
        target = (tf.math.maximum(0.0,TPM) - gene_mean) / gene_std
    elif output_type == 'logTPM_uqn':
        target = log2(1.0 + tf.math.maximum(0.0,TPM_uqn))
    else:
        raise ValueError('input an appropriate input type')

        
    return {
        'inputs': tf.ensure_shape(inputs,[input_length,5]),
        'atac': tf.ensure_shape(atac, [input_length,1]),
        'target': tf.transpose(tf.reshape(target,[-1])),
        'TF_acc': TF_acc
    }


def deserialize_val(serialized_example, input_length, 
                    num_TFs,max_shift,output_type):
    """
    Deserialize bytes stored in TFRecordFile.
    """
    feature_map = {
        'atac': tf.io.FixedLenFeature([], tf.string),
        'exons': tf.io.FixedLenFeature([],tf.string),
        'sequence': tf.io.FixedLenFeature([],tf.string),
        'TPM': tf.io.FixedLenFeature([],tf.string),
        'TPM_uqn': tf.io.FixedLenFeature([],tf.string),
        'cell_type': tf.io.FixedLenFeature([],tf.string),
        'gene_mean': tf.io.FixedLenFeature([],tf.string),
        'gene_encoded': tf.io.FixedLenFeature([],tf.string),
        'gene_std': tf.io.FixedLenFeature([],tf.string),
        'TF_acc': tf.io.FixedLenFeature([], tf.string),
        'tss_tokens': tf.io.FixedLenFeature([], tf.string),
        'interval': tf.io.FixedLenFeature([],tf.string)
    }
    data = tf.io.parse_example(serialized_example, feature_map)

    ### stochastic sequence shift and gaussian noise
    shift = 500#random.randrange(0,max_shift,1)
    input_seq_length = input_length + max_shift
    interval_end = input_length + shift
    
    exons = tf.ensure_shape(tf.io.parse_tensor(data['exons'],
                                              out_type=tf.int32),
                            [input_seq_length,])
    exons = tf.cast(tf.slice(exons, [shift],[input_length]),dtype=tf.float32)
    
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                              out_type=tf.float32),
                           [input_seq_length,])
    atac = tf.slice(atac, [shift],[input_length])
    atac=tf.expand_dims(atac,1)

    sequence = one_hot(tf.strings.substr(data['sequence'],
                                         shift,input_length))
    
    TF_acc = tf.ensure_shape(tf.io.parse_tensor(data['TF_acc'],
                                              out_type=tf.float32),
                             [num_TFs,])
    
    inputs = tf.concat([tf.expand_dims(exons,1), sequence], axis=1)
    #inputs = tf.concat([tf.expand_dims(exons,1), inputs], axis=1)
    
    TPM = tf.io.parse_tensor(data['TPM'],out_type=tf.float32)
    TPM_uqn = tf.io.parse_tensor(data['TPM_uqn'],out_type=tf.float32)
    gene_mean = tf.io.parse_tensor(data['gene_mean'],out_type=tf.float32)
    gene_std = tf.io.parse_tensor(data['gene_std'],out_type=tf.float32)
    #print(tf.io.parse_tensor(data['cell_type'],out_type=tf.int32))s
    cell_type = tf.io.parse_tensor(data['cell_type'],out_type=tf.int32)
    #print(data['cell_type'])
    gene_encoded = tf.io.parse_tensor(data['gene_encoded'],out_type=tf.int32)

    #print(TPM)
    if output_type == 'logTPM':
        target = log2(1.0 + tf.math.maximum(0.0,TPM))
    elif output_type == 'zTPM':
        target = (tf.math.maximum(0.0,TPM) - gene_mean) / gene_std
    elif output_type == 'logTPM_uqn':
        target = log2(1.0 + tf.math.maximum(0.0,TPM_uqn))
    else:
        raise ValueError('input an appropriate input type')
    
    tss_tokens = tf.ensure_shape(tf.io.parse_tensor(data['tss_tokens'],
                                              out_type=tf.int32),
                            [input_seq_length,])
    tss_tokens = tf.slice(tss_tokens, [shift],[input_length])
        
    return {
        'inputs': tf.ensure_shape(inputs,[input_length,5]),
        'atac': tf.ensure_shape(atac, [input_length,1]),
        'target': tf.reshape(target,[-1]),
        'TF_acc': TF_acc,
        'cell_type': tf.transpose(tf.reshape(cell_type,[-1])),
        'gene_encoded': tf.transpose(tf.reshape(gene_encoded,[-1])),
        'tss_tokens': tf.ensure_shape(tss_tokens,[input_length])
        #'interval': data['interval']
    }
                    
def return_dataset(gcs_path,
                   split,
                   organism,
                   batch,
                   input_length,
                   max_shift,
                   output_type,
                   options,
                   num_parallel,
                   num_epoch,
                   num_TFs):
    """
    return a tf dataset object for given gcs path
    """
    wc = str(organism) + "*.tfr"
    
    list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                split,
                                                wc)))
    #print(os.path.join(gcs_path,split,wc))
    
    random.shuffle(list_files)
    files = tf.data.Dataset.list_files(list_files)
    
    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      buffer_size=10000000,
                                      num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)

    dataset = dataset.map(lambda record: deserialize(record,
                                                     input_length,
                                                     num_TFs,
                                                     max_shift,
                                                     output_type),
                          deterministic=False,
                          num_parallel_calls=num_parallel)
    
    
    return dataset.repeat(num_epoch).batch(batch,drop_remainder=True).prefetch(1)


def return_dataset_val(gcs_path,
                       split,
                       organism,
                       batch,
                       input_length,
                       max_shift,
                       output_type,
                       options,
                       num_parallel,
                       num_epoch,
                       num_TFs):
    """
    return a tf dataset object for given gcs path
    """

    wc = str(organism) + "*.tfr"
    
    list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                split,
                                                wc)))

    random.shuffle(list_files)
    files = tf.data.Dataset.list_files(list_files)

    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      buffer_size=100000,
                                      num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)

    dataset = dataset.map(lambda record: deserialize_val(record,
                                                         input_length,
                                                         num_TFs,
                                                         max_shift,
                                                         output_type),
                          deterministic=False,
                          num_parallel_calls=num_parallel)


    return dataset.repeat(num_epoch).batch(batch, drop_remainder=True).prefetch(1)

def return_distributed_iterators(heads_dict,
                                 gcs_path,
                                 global_batch_size,
                                 input_length,
                                 max_shift,
                                 output_type,
                                 num_parallel_calls,
                                 num_epoch,
                                 strategy,
                                 options,
                                 num_TFs):
    """ 
    returns train + val dictionaries of distributed iterators
    for given heads_dictionary
    """
    with strategy.scope():
        
        data_it_tr_list = []
        data_it_val_list = []

        for org,index in heads_dict.items():
            tr_data = return_dataset(gcs_path,
                                     "train",org, 
                                     global_batch_size,
                                     input_length,
                                     max_shift,
                                     output_type,
                                     options,
                                     num_parallel_calls,
                                     num_epoch,
                                     num_TFs)
            
            val_data = return_dataset_val(gcs_path,
                                         "val",org, 
                                         global_batch_size,
                                         input_length,
                                         max_shift,
                                         output_type,
                                         options,
                                         num_parallel_calls,
                                         num_epoch,
                                         num_TFs)
            


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
                   current_pearsons,
                   logged_pearsons,
                   current_epoch,
                   best_epoch,
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
    try: 
        best_loss = min(logged_val_losses[:-1])
        best_pearsons=max(logged_pearsons[:-1])
        
    except ValueError:
        best_loss = current_val_loss
        best_pearsons = current_pearsons
        
    stop_criteria = False
    ## if min delta satisfied then log loss
    
    if (current_val_loss >= (best_loss - min_delta)):# and (current_pearsons <= best_pearsons):
        patience_counter += 1
        if patience_counter >= patience:
            stop_criteria=True
    else:

        best_epoch = np.argmin(logged_val_losses)
        ## save current model
        if (current_epoch % save_freq) == 0:
            print('Saving model...')
            model_name = save_directory + "/" + \
                            saved_model_basename + "/iteration_" + \
                                str(current_epoch) + "/saved_model"
            model.save_weights(model_name)
            ### write to logging file in saved model dir to model parameters and current epoch info
            
        patience_counter = 0
        stop_criteria = False
    
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
    parser.add_argument('--max_shift',
                        dest='max_shift',
                        type=int,
                        help= 'max_shift')
    parser.add_argument('--target_unit',
                        dest='target_unit',
                        help= 'target_unit')
    
    
    ## training loop parameters
    parser.add_argument('--batch_size', dest = 'batch_size',
                        type=int, help='batch_size')
    parser.add_argument('--num_epochs', dest = 'num_epochs',
                        type=int, help='num_epochs')
    parser.add_argument('--warmup_frac', dest = 'warmup_frac',
                        type=float, help='warmup_frac')
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
    parser.add_argument('--min_lr',
                        dest='min_lr',
                        help= 'min_lr')
    parser.add_argument('--epsilon',
                        dest='epsilon',
                        default=1.0e-08,
                        type=float,
                        help= 'epsilon')
    parser.add_argument('--rectify',
                        dest='rectify',
                        default=True,
                        help= 'rectify')
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
    parser.add_argument('--weight_decay_frac',
                        dest='weight_decay_frac',
                        type=str,
                        help= 'weight_decay_frac')
    parser.add_argument('--sync_period',
                        type=int,
                        dest='sync_period',
                        help= 'sync_period')
    parser.add_argument('--slow_step_frac',
                        type=float,
                        dest='slow_step_frac',
                        help= 'slow_step_frac')
    
    
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
    parser.add_argument('--conv_filter_size_1',
                        dest='conv_filter_size_1',
                        help= 'conv_filter_size_1')
    parser.add_argument('--conv_filter_size_2',
                        dest='conv_filter_size_2',
                        help= 'conv_filter_size_2')
    parser.add_argument('--dim',
                        dest='dim',
                        type=int,
                        help= 'mask_pos_dim')
    parser.add_argument('--max_seq_length',
                        dest='max_seq_length',
                        type=int,
                        help= 'max_seq_length')
    parser.add_argument('--rel_pos_bins',
                        dest='rel_pos_bins',
                        type=int,
                        help= 'rel_pos_bins')
    parser.add_argument('--kernel_regularizer',
                        dest='kernel_regularizer',
                        type=float,
                        help= 'kernel_regularizer')
    parser.add_argument('--savefreq',
                        dest='savefreq',
                        type=int,
                        help= 'savefreq')
    parser.add_argument('--use_rot_emb',
                        dest='use_rot_emb',
                        help= 'use_rot_emb')
    parser.add_argument('--use_mask_pos',
                        dest='use_mask_pos',
                        help= 'use_mask_pos')



    args = parser.parse_args()
    return parser
    
    
    
def one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    vocabulary = tf.constant(['A', 'C', 'G', 'T'])
    mapping = tf.constant([0, 1, 2, 3])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=0)

    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))

    out = tf.one_hot(table.lookup(input_characters), 
                      depth = 4, 
                      dtype=tf.float32)
    return out

def rev_comp_one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))
    input_characters = tf.reverse(input_characters,[0])
    
    vocabulary = tf.constant(['T', 'G', 'C', 'A'])
    mapping = tf.constant([0, 1, 2, 3])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=0)

    out = tf.one_hot(table.lookup(input_characters), 
                      depth = 4, 
                      dtype=tf.float32)
    return out


def cell_type_parser(input_str, vocabulary, mapping):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''

    init_celltype = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                                         values=mapping)
    table_celltype = tf.lookup.StaticHashTable(init_celltype, default_value=0)

    input_characters = input_str

    return table_celltype.lookup(input_characters)

def parse_gene_map(gene_map_file):
    input_df = pd.read_csv(gene_map_file,sep='\t')
    
    input_df.columns = ['gene', 'transcript',
                        'protein_type', 'gene_symbol']
    
    input_df['gene'] = input_df['gene'].map(lambda a: a.split('.')[0])
    input_df['transcript'] = input_df['transcript'].map(lambda a: a.split('.')[0])
    
    out_list = list(set(list(input_df['gene'].tolist())))
    
    return tf.constant(['empty'] + out_list, dtype=tf.string), tf.range(len(out_list) + 1)

    

def feature_parser(input_feature_map,
                   local_feature_info,
                   table_global):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    
    
    parsed_feature_map = tf.strings.split(input_feature_map, sep = ';')
    
    def choose_one(b):
        split_b = tf.strings.split(b,sep=',')
        def f1(): return split_b[0]
        def f2(): return b
        #def f1(): return tf.strings.regex_replace(b, ",", ".")
        #def f2(): return b
        return tf.cond(tf.size(split_b) > 1, f1,f2)
    
    feature_map_split = tf.map_fn(choose_one, parsed_feature_map)

    parsed_feature_info = tf.strings.split(local_feature_info, sep = ';')
    
    def return_first(b):
        split_b = tf.strings.split(b,sep=':')
        return split_b[0]
    
    local_gene_name = tf.vectorized_map(return_first, parsed_feature_info)
    
    def return_gene_encoding(b):
        def f1(): 
            return local_gene_name[b-1]
        def f2(): 
            return tf.constant(b'empty',dtype=tf.string)
        return tf.cond(b > 0, f1, f2)
            
    converted = tf.map_fn(return_gene_encoding,
                          tf.strings.to_number(feature_map_split, tf.int32),
                          fn_output_signature=tf.string)
    
    #converted = tf.keras.layers.StringLookup(vocabulary=local_gene_encoding)()
    return_val = table_global.lookup(converted) 
    
    return return_val

    """
    if unit == 'gene':
        feature_map = data['genes_to_bin_map']
    else:
        feature_map = data['txs_to_bin_map']

    feature_map_split = tf.strings.split(feature_map, sep = ';')
    
    def choose_one(b):
        split_b = tf.strings.split(b,sep=',')
        #print(tf.size(split_b))
        def f1(): return split_b[0]
        def f2() : return b

        return tf.cond(tf.size(split_b) > 1, f1,f2)
    
    feature_map_split_float = tf.vectorized_map(lambda t: tf.strings.to_number(t,out_type=tf.float32), 
                                        tf.map_fn(choose_one, feature_map_split))
    """
    
    
    
def make_plots(y_trues,y_preds, cell_types,gene_map, file_name_prefix):
    unique_preds = {}
    unique_trues = {}
    for k,x in enumerate(gene_map):
        unique_preds[(cell_types[k],x)] = y_preds[k]
        unique_trues[(cell_types[k],x)] = y_trues[k]


    unique_preds = dict(sorted(unique_preds.items()))
    unique_trues = dict(sorted(unique_trues.items()))

    overall_gene_level_corr = pearsonr(y_trues,
                                       y_preds)[0]

    fig_gene_level,ax_gene_level=plt.subplots(figsize=(6,6))
    data = np.vstack([y_trues,y_preds])
    kernel = stats.gaussian_kde(data)(data)

    sns.scatterplot(
        x=y_trues,
        y=y_preds,
        c=kernel,
        cmap="viridis")
    
    plt.xlabel("true log2(1.0+TPM)")
    plt.ylabel("predicted log2(1.0+TPM)")
    plt.xlim(0, max(y_trues))
    plt.ylim(0, max(y_trues))
    plt.title("overall correlation, all genes, all cells")
    ### now compute correlations across cell types
    across_cells_preds = {}
    across_cells_trues = {}
    
    
    low_y_true_indices = np.where(y_trues < 3.5)
    low_y_trues = y_trues[low_y_true_indices]
    low_y_preds = y_preds[low_y_true_indices]
    
    low_gene_level_corr = pearsonr(low_y_trues,
                                       low_y_preds)[0]
    
    fig_gene_level_l,ax_gene_level_l=plt.subplots(figsize=(6,6))
    data = np.vstack([low_y_trues,low_y_preds])
    kernel = stats.gaussian_kde(data)(data)

    sns.scatterplot(
        x=low_y_trues,
        y=low_y_preds,
        c=kernel,
        cmap="viridis")
    
    plt.xlabel("true log2(1.0+TPM)")
    plt.ylabel("predicted log2(1.0+TPM)")
    plt.xlim(0, max(low_y_trues))
    plt.ylim(0, max(low_y_trues))
    plt.title("correlation LOW expression, all genes, all cells")
    
    high_y_true_indices = np.where(y_trues >= 3.5)
    high_y_trues = y_trues[high_y_true_indices]
    high_y_preds = y_preds[high_y_true_indices]
    
    high_gene_level_corr = pearsonr(high_y_trues,
                                    high_y_preds)[0]
    
    fig_gene_level_h,ax_gene_level_h=plt.subplots(figsize=(6,6))
    data = np.vstack([high_y_trues,high_y_preds])
    kernel = stats.gaussian_kde(data)(data)

    sns.scatterplot(
        x=high_y_trues,
        y=high_y_preds,
        c=kernel,
        cmap="viridis")
    
    plt.xlabel("true log2(1.0+TPM)")
    plt.ylabel("predicted log2(1.0+TPM)")
    plt.xlim(0, max(high_y_trues))
    plt.ylim(0, max(high_y_trues))
    plt.title("correlation HIGH expression, all genes, all cells")
    
    ### now compute correlations across cell types
    across_cells_preds = {}
    across_cells_trues = {}
    
    for k,v in unique_preds.items():
        cell_t,gene_name = k
        if cell_t not in across_cells_preds.keys():
            across_cells_preds[cell_t] = []
            across_cells_trues[cell_t] = []
        else:
            across_cells_preds[cell_t].append(v)
            across_cells_trues[cell_t].append(unique_trues[k])
    cell_specific_corrs = []
    for k,v in across_cells_preds.items():
        trues = []
        preds = []
        for idx,x in enumerate(v):
            #if len(x) > 0:
            preds.append(x)
            trues.append(across_cells_trues[k][idx])
        try: 
            cell_specific_corrs.append(pearsonr(trues, 
                                                preds)[0])
        except np.linalg.LinAlgError:
            continue
        except ValueError:
            continue

    fig_cell_spec,ax_cell_spec=plt.subplots(figsize=(6,6))
    sns.histplot(x=np.asarray(cell_specific_corrs), bins=50)
    plt.xlabel("single cell-type cross gene correlations")
    plt.ylabel("count")
    plt.title("log-log pearsonsR")
    cell_spec_median = np.nanmedian(cell_specific_corrs)


    ### now compute correlations across genes
    across_genes_preds = {}
    across_genes_trues = {}

    for k,v in unique_preds.items():
        cell_t,gene_name = k
        if gene_name not in across_genes_preds.keys():
            across_genes_preds[gene_name] = []
            across_genes_trues[gene_name] = []
        else:
            across_genes_preds[gene_name].append(v)
            across_genes_trues[gene_name].append(unique_trues[k])
    genes_specific_corrs = []
    genes_specific_vars = []
    for k,v in across_genes_preds.items():
        trues = []
        preds = []
        for idx, x in enumerate(v):
            #if len(x) > 0:
            preds.append(x)
            trues.append(across_genes_trues[k][idx])
        try: 
            genes_specific_corrs.append(pearsonr(trues, 
                                                 preds)[0])
            genes_specific_vars.append(np.nanstd(trues))
        except np.linalg.LinAlgError:
            continue
        except ValueError:
            continue
            
    fig_gene_spec,ax_gene_spec=plt.subplots(figsize=(6,6))
    sns.histplot(x=np.asarray(genes_specific_corrs), bins=50)
    plt.xlabel("single gene cross cell-type correlations")
    plt.ylabel("count")
    plt.title("log-log pearsonsR")
    gene_spec_median_corr = np.nanmedian(genes_specific_corrs)
    
    
    file_name = file_name_prefix + ".val.out.tsv"
    
    dataset = pd.DataFrame({'true': y_trues, 
                            'preds': y_preds,
                            'cell_types': cell_types,
                            'gene_map': gene_map})
    
    dataset.to_csv(file_name, sep='\t',index=False)
    
    return overall_gene_level_corr,low_gene_level_corr, high_gene_level_corr,cell_spec_median,gene_spec_median_corr, fig_gene_level,fig_gene_level_l,fig_gene_level_h, fig_cell_spec,fig_gene_spec






def deserialize_interpret(serialized_example, input_length, 
                    output_length,num_TFs,max_shift,output_type):
    """
    Deserialize bytes stored in TFRecordFile.
    """
    feature_map = {
        'atac': tf.io.FixedLenFeature([], tf.string),
        'exons': tf.io.FixedLenFeature([],tf.string),
        'sequence': tf.io.FixedLenFeature([],tf.string),
        'TPM': tf.io.FixedLenFeature([],tf.string),
        'TPM_uqn': tf.io.FixedLenFeature([],tf.string),
        'cell_type': tf.io.FixedLenFeature([],tf.string),
        'gene_mean': tf.io.FixedLenFeature([],tf.string),
        'gene_encoded': tf.io.FixedLenFeature([],tf.string),
        'gene_std': tf.io.FixedLenFeature([],tf.string),
        'TF_acc': tf.io.FixedLenFeature([], tf.string),
        'tss_tokens': tf.io.FixedLenFeature([], tf.string),
        'interval': tf.io.FixedLenFeature([],tf.string)
    }
    data = tf.io.parse_example(serialized_example, feature_map)

    ### stochastic sequence shift and gaussian noise
    shift = 500#random.randrange(0,max_shift,1)
    input_seq_length = input_length + max_shift
    interval_end = input_length + shift
    
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                              out_type=tf.float32),
                           [input_seq_length,])
    atac = tf.slice(atac, [shift],[input_length])
    
    exons = tf.ensure_shape(tf.io.parse_tensor(data['exons'],
                                              out_type=tf.int32),
                            [input_seq_length,])
    exons = tf.cast(tf.slice(exons, [shift],[input_length]),dtype=tf.float32)
    
    sequence = one_hot(tf.strings.substr(data['sequence'],
                                         shift,input_length))
    
    TF_acc = tf.ensure_shape(tf.io.parse_tensor(data['TF_acc'],
                                              out_type=tf.float32),
                             [num_TFs,])
    
    inputs = tf.concat([tf.expand_dims(atac, 1), sequence], axis=1)
    inputs = tf.concat([tf.expand_dims(exons,1), inputs], axis=1)
    
    TPM = tf.io.parse_tensor(data['TPM'],out_type=tf.float32)
    TPM_uqn = tf.io.parse_tensor(data['TPM_uqn'],out_type=tf.float32) 
    if output_type == 'logTPM':
        target = log10(1.0 + TPM)
    elif output_type == 'zTPM':
        gene_mean = tf.io.parse_tensor(data['gene_mean'],out_type=tf.float32)
        gene_std = tf.io.parse_tensor(data['gene_std'],out_type=tf.float32)
        target = (TPM - gene_mean) / gene_std
    elif output_type == 'logTPM_uqn':
        target = log10(1.0 + TPM_uqn)
    else:
        raise ValueError('input an appropriate input type')
    
    tss_tokens = tf.ensure_shape(tf.io.parse_tensor(data['tss_tokens'],
                                              out_type=tf.int32),
                            [input_seq_length,])
    tss_tokens = tf.slice(tss_tokens, [shift],[input_length])
        
    return {
        'inputs': inputs,
        'target': target,
        'TF_acc': TF_acc,
        'cell_type': tf.io.parse_tensor(data['cell_type'],out_type=tf.int32),
        'gene_encoded': tf.io.parse_tensor(data['gene_encoded'],out_type=tf.int32),
        'tss_tokens': tss_tokens,
        'interval': data['interval']
    }

def return_dataset_interpret(gcs_path,
                             input_length,
                             output_length_pre,
                             crop_size,
                             out_length,
                             options,
                             num_parallel,
                             num_epoch,
                             seed,
                             num_TFs):
    """
    return a tf dataset object for given gcs path
    """
    
    list_files = tf.io.gfile.glob(gcs_path)

    files = tf.data.Dataset.list_files(list_files)

    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      buffer_size=100000,
                                      num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)

    dataset = dataset.map(lambda record: deserialize_interpret(record,
                                                               input_length,
                                                               output_length_pre,
                                                               crop_size,
                                                               out_length,
                                                               seed,
                                                               num_TFs),
                          deterministic=False,
                          num_parallel_calls=num_parallel)


    return dataset.repeat(num_epoch).batch(1, drop_remainder=True).prefetch(1)


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def log2(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator




def sum_log(x):
    return np.log10(1.0 + np.nansum(x))


def fourier_att_prior_loss(
    output, input_grads, freq_limit=150, limit_softness=0.2,
    att_prior_grad_smooth_sigma=3
):
    """
    Computes an attribution prior loss for some given training examples,
    using a Fourier transform form.
    Arguments:
        `output`: a B-tensor, where B is the batch size; each entry is a
            predicted logTPM value
        `input_grads`: a B x L x 4 tensor, where B is the batch size, L is
            the length of the input; this needs to be the gradients of the
            input with respect to the output; this should be
            *gradient times input*
        `freq_limit`: the maximum integer frequency index, k, to consider for
            the loss; this corresponds to a frequency cut-off of pi * k / L;
            k should be less than L / 2
        `limit_softness`: amount to soften the limit by, using a hill
            function; None means no softness
        `att_prior_grad_smooth_sigma`: amount to smooth the gradient before
            computing the loss
    Returns a single scalar Tensor consisting of the attribution loss for
    the batch.
    """
    abs_grads = kb.sum(kb.abs(input_grads), axis=2)

    # Smooth the gradients
    grads_smooth = smooth_tensor_1d(
        abs_grads, att_prior_grad_smooth_sigma
    )
    
    # Only do the positives
    #pos_grads = grads_smooth[status == 1]

    #if pos_grads.numpy().size:
    pos_fft = tf.signal.rfft(tf.cast(abs_grads,dtype=tf.float32))
    pos_mags = tf.abs(pos_fft)
    pos_mag_sum = kb.sum(pos_mags, axis=1, keepdims=True)
    zero_mask = tf.cast(pos_mag_sum == 0, tf.float32)
    pos_mag_sum = pos_mag_sum + zero_mask  # Keep 0s when the sum is 0  
    pos_mags = pos_mags / pos_mag_sum

    # Cut off DC
    pos_mags = pos_mags[:, 1:]

    # Construct weight vector
    if limit_softness is None:
        weights = tf.sequence_mask(
            [freq_limit], maxlen=tf.shape(pos_mags)[1], dtype=tf.float32
        )
    else:
        weights = tf.sequence_mask(
            [freq_limit], maxlen=tf.shape(pos_mags)[1], dtype=tf.float32
        )
        x = tf.abs(tf.range(
            -freq_limit + 1, tf.shape(pos_mags)[1] - freq_limit + 1, dtype=tf.float32
        ))  # Take absolute value of negatives just to avoid NaN; they'll be removed
        decay = 1 / (1 + tf.pow(x, limit_softness))
        weights = weights + ((1.0 - weights) * decay)

    # Multiply frequency magnitudes by weights
    pos_weighted_mags = pos_mags * weights

    # Add up along frequency axis to get score
    pos_score = tf.reduce_sum(pos_weighted_mags, axis=1)
    pos_loss = 1 - pos_score
    print(pos_loss)
    return tf.reduce_mean(pos_loss)
    
    
    
    
def smooth_tensor_1d(input_tensor, smooth_sigma):
    """
    Smooths an input tensor along a dimension using a Gaussian filter.
    Arguments:
        `input_tensor`: a A x B tensor to smooth along the second dimension
        `smooth_sigma`: width of the Gaussian to use for smoothing; this is the
            standard deviation of the Gaussian to use, and the Gaussian will be
            truncated after 1 sigma (i.e. the smoothing window is
            1 + (2 * sigma); sigma of 0 means no smoothing
    Returns an array the same shape as the input tensor, with the dimension of
    `B` smoothed.
    """
    input_tensor = tf.cast(input_tensor,dtype=tf.float32)
    # Generate the kernel
    if smooth_sigma == 0:
        sigma, truncate = 1, 0
    else:
        sigma, truncate = smooth_sigma, 1
    base = np.zeros(1 + (2 * sigma))
    base[sigma] = 1  # Center of window is 1 everywhere else is 0
    kernel = scipy.ndimage.gaussian_filter(base, 
                                           sigma=sigma, 
                                           truncate=truncate)
    kernel = tf.constant(kernel,dtype=tf.float32)

    # Expand the input and kernel to 3D, with channels of 1
    input_tensor = tf.expand_dims(input_tensor, axis=2)  # Shape: A x B x 1
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=1), axis=2)  # Shape: (1 + 2s) x 1 x 1

    smoothed = tf.nn.conv1d(
        input_tensor, kernel, stride=1, padding="SAME", data_format="NWC"
    )
    print(smoothed)
    return tf.squeeze(smoothed, axis=2)