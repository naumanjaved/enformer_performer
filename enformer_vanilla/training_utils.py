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
#import logging
#from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision
import metrics as metrics ## switch to src 
import optimizers
import schedulers
import pandas as pd
import seaborn as sns
from scipy.stats.stats import pearsonr, spearmanr
from scipy.stats import linregress
from scipy import stats
import keras.backend as kb

import scipy.special
import scipy.stats
import scipy.ndimage

import numpy as np
from sklearn import metrics as sklearn_metrics

from tensorflow.keras import initializers as inits
from scipy.stats import zscore

tf.keras.backend.set_floatx('float32')

def tf_tpu_initialize(tpu_name,zone):
    """Initialize TPU and return global batch size for loss calculation
    Args:
        tpu_name
    Returns:
        distributed strategy
    """
    
    try: 
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_name,zone=zone)
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)

    except ValueError: # no TPU found, detect GPUs
        strategy = tf.distribute.get_strategy()

    return strategy


def get_initializers(checkpoint_path):
    
    inside_checkpoint=tf.train.list_variables(tf.train.latest_checkpoint(checkpoint_path))
    reader = tf.train.load_checkpoint(checkpoint_path)

    initializers_dict = {'stem_conv_k': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/0/w/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_conv_b': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/0/b/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_k': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_b': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_g': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_b': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_m': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE')[0,0:]),
                         'stem_res_conv_BN_v': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE')[0,0:]),
                         'stem_pool': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/2/_logit_linear/w/.ATTRIBUTES/VARIABLE_VALUE'))}

    for i in range(6):
        var_name_stem = 'module/_trunk/_layers/1/_layers/' + str(i) + '/_layers/' #0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE'

        conv1_k = var_name_stem + '0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE'
        conv1_b = var_name_stem + '0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_g = var_name_stem + '0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_b = var_name_stem + '0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_m = var_name_stem + '0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_v = var_name_stem + '0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE'
        conv2_k = var_name_stem + '1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE'
        conv2_b = var_name_stem + '1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_g = var_name_stem + '1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_b = var_name_stem + '1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_m = var_name_stem + '1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_v = var_name_stem + '1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE'
        pool = var_name_stem + '2/_logit_linear/w/.ATTRIBUTES/VARIABLE_VALUE'
        all_vars = [conv1_k,
                    conv1_b,
                    BN1_g,
                    BN1_b,
                    BN1_m,
                    BN1_v,
                    conv2_k,
                    conv2_b,
                    BN2_g,
                    BN2_b,
                    BN2_m,
                    BN2_v,
                    pool]

        out_dict = {'conv1_k_' + str(i): inits.Constant(reader.get_tensor(conv1_k)),
                    'conv1_b_' + str(i): inits.Constant(reader.get_tensor(conv1_b)),
                    'BN1_g_' + str(i): inits.Constant(reader.get_tensor(BN1_g)),
                    'BN1_b_' + str(i): inits.Constant(reader.get_tensor(BN1_b)),
                    'BN1_m_' + str(i): inits.Constant(reader.get_tensor(BN1_m)[0,0:]),
                    'BN1_v_' + str(i): inits.Constant(reader.get_tensor(BN1_v)[0,0:]),
                    'conv2_k_' + str(i): inits.Constant(reader.get_tensor(conv2_k)),
                    'conv2_b_' + str(i): inits.Constant(reader.get_tensor(conv2_b)),
                    'BN2_g_' + str(i): inits.Constant(reader.get_tensor(BN2_g)),
                    'BN2_b_' + str(i): inits.Constant(reader.get_tensor(BN2_b)),
                    'BN2_m_' + str(i): inits.Constant(reader.get_tensor(BN2_m)[0,0:]),
                    'BN2_v_' + str(i): inits.Constant(reader.get_tensor(BN2_v)[0,0:]),
                    'pool_' + str(i): inits.Constant(reader.get_tensor(pool))}
        initializers_dict.update(out_dict)
    return initializers_dict

def return_train_val_functions(model,
                               train_steps_human,
                               val_steps_human,
                               val_steps_mouse,
                               val_steps_TSS,
                               optimizer,
                               strategy,
                               metric_dict,
                               global_batch_size,
                               gradient_clip,
                               loss_fn_main='poisson'):
    
    
    if loss_fn_main == 'mse':
        loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    elif loss_fn_main == 'poisson':
        loss_fn = tf.keras.losses.Poisson(reduction=tf.keras.losses.Reduction.NONE)
    else:
        raise ValueError('loss_fn_not_implemented')


    metric_dict["hg_corr_stats"] = metrics.correlation_stats_gene_centered(name='hg_corr_stats')
    metric_dict["human_tr"] = tf.keras.metrics.Mean("human_tr_loss",
                                                 dtype=tf.float32)
    metric_dict["human_val"] = tf.keras.metrics.Mean("human_val_loss",
                                                  dtype=tf.float32)
    metric_dict['human_pearsonsR'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
    metric_dict['human_R2'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})
    
    metric_dict["mm_corr_stats"] = metrics.correlation_stats_gene_centered(name='mm_corr_stats')
    metric_dict["mouse_tr"] = tf.keras.metrics.Mean("mouse_tr_loss",
                                                 dtype=tf.float32)
    metric_dict["mouse_val"] = tf.keras.metrics.Mean("mouse_val_loss",
                                                  dtype=tf.float32)
    metric_dict['mouse_pearsonsR'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
    metric_dict['mouse_R2'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})

    #train_steps, val_steps = steps_tuple

    
    def dist_train_step_both(iter_human,iter_mouse):    
        @tf.function(jit_compile=True)
        def train_step_h(inputs):
            sequence=tf.cast(inputs['sequence'],dtype=tf.float32)
            target=tf.cast(inputs['target'],dtype=tf.float32)

            with tf.GradientTape() as tape:
                output = model(sequence,
                               is_training=True)['human']

                output = tf.cast(output,dtype=tf.float32)
                loss = tf.math.reduce_mean(loss_fn(target,output)) * (1. / global_batch_size)

            gradients = tape.gradient(loss, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 
                                                  gradient_clip)
            optimizer.apply_gradients(zip(gradients, 
                                           model.trainable_variables))
            metric_dict["human_tr"].update_state(loss)
            
        @tf.function(jit_compile=True)
        def train_step_m(inputs):
            sequence=tf.cast(inputs['sequence'],dtype=tf.float32)
            target=tf.cast(inputs['target'],dtype=tf.float32)

            with tf.GradientTape() as tape:
                output = model(sequence,
                               is_training=True)["mouse"]

                output = tf.cast(output,dtype=tf.float32)
                loss = tf.math.reduce_mean(loss_fn(target,output)) * (1. / global_batch_size)

            gradients = tape.gradient(loss, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 
                                                  gradient_clip)
            optimizer.apply_gradients(zip(gradients, 
                                           model.trainable_variables))
            
            metric_dict["mouse_tr"].update_state(loss)
            
        for _ in tf.range(train_steps_human):
            strategy.run(train_step_h,
                         args=(next(iter_human),))
            strategy.run(train_step_m,
                         args=(next(iter_mouse),))
            
    def dist_train_step_h(iter_human):    
        @tf.function(jit_compile=True)
        def train_step_h(inputs):
            sequence=tf.cast(inputs['sequence'],dtype=tf.float32)
            target=tf.cast(inputs['target'],dtype=tf.float32)

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                conv_vars = model.trunk.submodules[0].trainable_variables + \
                                model.trunk.submodules[1].trainable_variables 
                remaining_vars = model.trunk.submodules[2].trainable_variables + \
                                    model.trunk.submodules[4].trainable_variables + \
                                        model.heads['human'].trainable_variables
                vars_all = conv_vars + remaining_vars
                for var in vars_all:
                    tape.watch(var)
                output = model(sequence,
                               is_training=True)['human']

                output = tf.cast(output,dtype=tf.float32)
                loss = tf.math.reduce_mean(loss_fn(target,output)) * (1. / global_batch_size)

            gradients = tape.gradient(loss, vars_all)
            gradients, _ = tf.clip_by_global_norm(gradients, 
                                                  gradient_clip)
            optimizer1.apply_gradients(zip(gradients[:len(conv_vars)], 
                                           conv_vars))
            optimizer2.apply_gradients(zip(gradients[len(conv_vars):], 
                                           remaining_vars))
            metric_dict["human_tr"].update_state(loss)
            
        for _ in tf.range(train_steps_human):
            strategy.run(train_step_h,
                         args=(next(iter_human),))

    def dist_val_step_h(iterator):
        @tf.function(jit_compile=True)
        def val_step(inputs):
            sequence=tf.cast(inputs['sequence'],dtype=tf.float32)
            target=tf.cast(inputs['target'],dtype=tf.float32)

            output = model(sequence,
                           is_training=False)['human']
            output = tf.cast(output,dtype=tf.float32)
            
            loss = tf.math.reduce_mean(loss_fn(target,output)) * (1. / global_batch_size)

            metric_dict["human_val"].update_state(loss)
            metric_dict['human_pearsonsR'].update_state(target, output)
            metric_dict['human_R2'].update_state(target, output)
        for _ in tf.range(val_steps_human): ## for loop within @tf.fuction for improved TPU performance
            strategy.run(val_step,
                         args=(next(iterator),))
        
    def dist_val_step_m(iterator):
        @tf.function(jit_compile=True)
        def val_step(inputs):
            sequence=tf.cast(inputs['sequence'],dtype=tf.float32)
            target=tf.cast(inputs['target'],dtype=tf.float32)

            output = model(sequence,
                           is_training=False)['mouse']
            
            output = tf.cast(output,dtype=tf.float32)
            
            loss = tf.math.reduce_mean(loss_fn(target,output)) * (1. / global_batch_size)

            metric_dict["mouse_val"].update_state(loss)
            metric_dict['mouse_pearsonsR'].update_state(target, output)
            metric_dict['mouse_R2'].update_state(target, output)
        for _ in tf.range(val_steps_mouse): ## for loop within @tf.fuction for improved TPU performance
            strategy.run(val_step,
                         args=(next(iterator),))
            
                
    def dist_val_step_TSS(iterator): #input_batch, model, optimizer, organism, gradient_clip):
        @tf.function(jit_compile=True)
        def val_step(inputs):
            target = inputs['target'][:,:,4675:]
                
            sequence=tf.cast(inputs['sequence'], dtype=tf.float32)

            tss_mask = tf.cast(inputs['tss_mask'],dtype=tf.float32)
            gene_name = inputs['gene_name']

            cell_types = inputs['cell_types']
            output = model(sequence,is_training=False)['human'][:,:,4675:]
            
            pred = tf.reduce_sum(tf.cast(output,dtype=tf.float32) * tss_mask,axis=1)
            true = tf.reduce_sum(tf.cast(target,dtype=tf.float32) * tss_mask,axis=1)
            
            return pred,true,gene_name,cell_types


        ta_pred = tf.TensorArray(tf.float32, size=0, dynamic_size=True) # tensor array to store preds
        ta_true = tf.TensorArray(tf.float32, size=0, dynamic_size=True) # tensor array to store vals
        ta_celltype = tf.TensorArray(tf.int32, size=0, dynamic_size=True) # tensor array to store preds
        ta_genemap = tf.TensorArray(tf.int32, size=0, dynamic_size=True)        

        for _ in tf.range(val_steps_TSS): ## for loop within @tf.fuction for improved TPU performance

            pred_rep, true_rep, gene_rep, cell_type_rep = strategy.run(val_step,
                                                                       args=(next(iterator),))
            
            pred_reshape = tf.reshape(strategy.gather(pred_rep, axis=0), [-1]) # reshape to 1D
            true_reshape = tf.reshape(strategy.gather(true_rep, axis=0), [-1])
            cell_type_reshape = tf.reshape(strategy.gather(cell_type_rep, axis=0), [-1])
            gene_map_reshape = tf.reshape(strategy.gather(gene_rep, axis=0), [-1])
            
            ta_pred = ta_pred.write(_, pred_reshape)
            ta_true = ta_true.write(_, true_reshape)
            ta_celltype = ta_celltype.write(_, cell_type_reshape)
            ta_genemap = ta_genemap.write(_, gene_map_reshape)
        metric_dict["hg_corr_stats"].update_state(ta_true.concat(),
                                                  ta_pred.concat(),
                                                  ta_celltype.concat(),
                                                  ta_genemap.concat())
        ta_true.close()
        ta_pred.close()
        ta_celltype.close()
        ta_genemap.close()

    def build_step(iterator): #input_batch, model, optimizer, organism, gradient_clip):
        @tf.function(jit_compile=True)
        def val_step(inputs):
            target=tf.cast(inputs['target'],
                           dtype = tf.float32)
            sequence=tf.cast(inputs['sequence'],
                             dtype=tf.float32)
            output = model(sequence, is_training=False)['human']

        for _ in tf.range(1): ## for loop within @tf.fuction for improved TPU performance
            strategy.run(val_step, args=(next(iterator),))
    

    return dist_train_step_both,dist_train_step_h, dist_val_step_h,dist_val_step_m,dist_val_step_TSS,build_step, metric_dict


def deserialize_tr(serialized_example,input_length,max_shift, num_targets,g):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
      'sequence': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.string),
    }
    ### stochastic sequence shift and gaussian noise

    rev_comp = tf.math.round(g.uniform([], 0, 1))

    shift = g.uniform(shape=(),
                      minval=0,
                      maxval=max_shift,
                      dtype=tf.int32)

    for k in range(max_shift):
        if k == shift:
            interval_end = input_length + k
            seq_shift = k
        else:
            seq_shift=0
    
    input_seq_length = input_length + max_shift


    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (input_length + max_shift, 4))
    sequence = tf.cast(sequence, tf.float32)
    sequence = tf.slice(sequence, [seq_shift,0],[input_length,-1])
    
    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (896, num_targets))
    
    if rev_comp == 1:
        sequence = tf.gather(sequence, [3, 2, 1, 0], axis=-1)
        sequence = tf.reverse(sequence, axis=[0])
        target = tf.reverse(target,axis=[0])
    
    return {'sequence': tf.ensure_shape(sequence,
                                        [input_length,4]),
            'target': tf.ensure_shape(target,
                                      [896,num_targets])}
                    


def deserialize_val(serialized_example,input_length,max_shift, num_targets):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
      'sequence': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.string)
    }
    
    data = tf.io.parse_example(serialized_example, feature_map)

    shift = 5
    input_seq_length = input_length + max_shift
    interval_end = input_length + shift
    
    ### rev_comp
    #rev_comp = random.randrange(0,2)

    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (input_length + max_shift, 4))
    sequence = tf.cast(sequence, tf.float32)
    sequence = tf.slice(sequence, [shift,0],[input_length,-1])
    
    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (896, num_targets))
    
    
    return {'sequence': tf.ensure_shape(sequence,
                                        [input_length,4]),
            'target': tf.ensure_shape(target,
                                      [896,num_targets])}


def deserialize_val_TSS(serialized_example,input_length,max_shift,num_targets):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
        'tss_mask': tf.io.FixedLenFeature([], tf.string),
        'gene_name': tf.io.FixedLenFeature([], tf.string)
    }
    
    shift = 5
    input_seq_length = input_length + max_shift
    interval_end = input_length + shift
    
    ### rev_comp
    #rev_comp = random.randrange(0,2)

    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (input_length + max_shift, 4))
    sequence = tf.cast(sequence, tf.float32)
    sequence = tf.slice(sequence, [shift,0],[input_length,-1])
    
    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (896, num_targets))
    
    tss_mask = tf.io.parse_tensor(example['tss_mask'],
                                  out_type=tf.int32)

    
    gene_name= tf.io.parse_tensor(example['gene_name'],out_type=tf.int32)
    gene_name = tf.tile(tf.expand_dims(gene_name,axis=0),[638])
    cell_types = tf.range(0,638)
    
    return {'sequence': tf.ensure_shape(sequence,
                                        [input_length,4]),
            'target': tf.ensure_shape(target,
                                      [896,num_targets]),
            'tss_mask': tf.ensure_shape(tss_mask,
                                        [896,1]),
            'gene_name': tf.ensure_shape(gene_name,
                                         [638,]),
            'cell_types': tf.ensure_shape(cell_types,
                                           [638,])}
                    
def return_dataset(gcs_path,
                   organism,
                   split,
                   tss_bool,
                   batch,
                   input_length,
                   max_shift,
                   num_targets,
                   options,
                   num_parallel,
                   num_epoch,
                   g):
    """
    return a tf dataset object for given gcs path
    """
    wc = str(split) + "*.tfr"
    if not tss_bool:
        list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                    organism,
                                                    "tfrecords",
                                                    wc)))
    else:
        list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                    wc)))

    files = tf.data.Dataset.list_files(list_files,shuffle=True, seed=42)
    
    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)
    if split == 'train':

        dataset = dataset.map(lambda record: deserialize_tr(record,
                                                            input_length,
                                                            max_shift,
                                                            num_targets,
                                                            g),
                              deterministic=False,
                              num_parallel_calls=num_parallel)
        
    else:
        if not tss_bool:
            dataset = dataset.map(lambda record: deserialize_val(record,
                                                             input_length,
                                                             max_shift,
                                                             num_targets),
                                  deterministic=False,
                                  num_parallel_calls=num_parallel)
        else:
            #print(list_files)
            dataset = dataset.map(lambda record: deserialize_val_TSS(record,
                                                             input_length,
                                                             max_shift,
                                                             num_targets),
                                  deterministic=False,
                                  num_parallel_calls=num_parallel)

    return dataset.repeat(num_epoch).batch(batch,drop_remainder=True).prefetch(1)



def return_distributed_iterators(gcs_paths_dict,
                                 gcs_path_tss,
                                 global_batch_size,
                                 input_length,
                                 max_shift,
                                 num_parallel_calls,
                                 num_epoch,
                                 strategy,
                                 options,
                                 g):
    """ 
    returns train + val dictionaries of distributed iterators
    for given heads_dictionary
    """
   # with strategy.scope():

    train_iters={}
    valid_iters={}
    for organism,organism_info in gcs_paths_dict.items():
        gcs_path,num_targets=organism_info
        tr_data = return_dataset(gcs_path,
                                 organism,
                                 "train",
                                 False,
                                 global_batch_size,
                                 input_length,
                                 max_shift,
                                 num_targets,
                                 options,
                                 num_parallel_calls,
                                 num_epoch,
                                 g)


        val_data = return_dataset(gcs_path,
                                  organism,
                                 "valid",
                                  False,
                                 global_batch_size,
                                 input_length,
                                 max_shift,
                                 num_targets,
                                 options,
                                 num_parallel_calls,
                                 num_epoch, 
                                 g)

        train_dist = strategy.experimental_distribute_dataset(tr_data)
        val_dist= strategy.experimental_distribute_dataset(val_data)


        tr_data_it = iter(train_dist)
        val_data_it = iter(val_dist)

        train_iters[organism]=tr_data_it
        valid_iters[organism]=val_data_it

    val_data_TSS = return_dataset(gcs_path_tss,
                                  "human",
                                 "valid",
                                 True,
                                 global_batch_size,
                                 input_length,
                                 max_shift,
                                 gcs_paths_dict['human'][1],
                                 options,
                                 num_parallel_calls,
                                 num_epoch,
                                  g)



    val_dist_TSS_dist= strategy.experimental_distribute_dataset(val_data_TSS)
    val_data_TSS_it = iter(val_dist_TSS_dist)

    return train_iters,valid_iters,val_data_TSS_it


def make_plots(y_trues,
               y_preds, 
               cell_types, 
               gene_map):

    results_df = pd.DataFrame()
    results_df['true'] = y_trues
    results_df['pred'] = y_preds
    results_df['gene_encoding'] =gene_map
    results_df['cell_type_encoding'] = cell_types
    
    results_df=results_df.groupby(['gene_encoding', 'cell_type_encoding']).agg({'true': 'sum', 'pred': 'sum'})
    results_df['true'] = np.log2(1.0+results_df['true'])
    results_df['pred'] = np.log2(1.0+results_df['pred'])
    
    results_df['true_zscore']=results_df.groupby(['cell_type_encoding']).true.transform(lambda x : zscore(x))
    results_df['pred_zscore']=results_df.groupby(['cell_type_encoding']).pred.transform(lambda x : zscore(x))
    
    true_zscore=results_df[['true_zscore']].to_numpy()[:,0]

    pred_zscore=results_df[['pred_zscore']].to_numpy()[:,0]

    try: 
        cell_specific_corrs=results_df.groupby('cell_type_encoding')[['true_zscore','pred_zscore']].corr(method='pearson').unstack().iloc[:,1].tolist()
    except np.linalg.LinAlgError as err:
        cell_specific_corrs = [0.0] * len(np.unique(cell_types))

    try: 
        gene_specific_corrs=results_df.groupby('gene_encoding')[['true_zscore','pred_zscore']].corr(method='pearson').unstack().iloc[:,1].tolist()
    except np.linalg.LinAlgError as err:
        gene_specific_corrs = [0.0] * len(np.unique(gene_map))
    
    corrs_overall = np.nanmean(cell_specific_corrs), \
                        np.nanmean(gene_specific_corrs)
                        
        
    fig_overall,ax_overall=plt.subplots(figsize=(6,6))
    
    ## scatter plot for 50k points max
    idx = np.random.choice(np.arange(len(y_trues)), 20000, replace=False)
    
    data = np.vstack([y_trues[idx],
                      y_preds[idx]])
    
    min_true = min(y_trues)
    max_true = max(y_trues)
    
    min_pred = min(y_preds)
    max_pred = max(y_preds)
    
    
    try:
        kernel = stats.gaussian_kde(data)(data)
        sns.scatterplot(
            x=y_trues[idx],
            y=y_preds[idx],
            c=kernel,
            cmap="viridis")
        ax_overall.set_xlim(min_true,max_true)
        ax_overall.set_ylim(min_pred,max_pred)
        plt.xlabel("log-true")
        plt.ylabel("log-pred")
        plt.title("overall gene corr")
    except np.linalg.LinAlgError as err:
        sns.scatterplot(
            x=y_trues[idx],
            y=y_preds[idx],
            cmap="viridis")
        ax_overall.set_xlim(min_true,max_true)
        ax_overall.set_ylim(min_pred,max_pred)
        plt.xlabel("log-true")
        plt.ylabel("log-pred")
        plt.title("overall gene corr")

    fig_gene_spec,ax_gene_spec=plt.subplots(figsize=(6,6))
    sns.histplot(x=np.asarray(gene_specific_corrs), bins=50)
    plt.xlabel("log-log pearsons")
    plt.ylabel("count")
    plt.title("single gene cross cell-type correlations")

    fig_cell_spec,ax_cell_spec=plt.subplots(figsize=(6,6))
    sns.histplot(x=np.asarray(cell_specific_corrs), bins=50)
    plt.xlabel("log-log pearsons")
    plt.ylabel("count")
    plt.title("single cell-type cross gene correlations")
        
        ### by coefficient variation breakdown
    figures = fig_cell_spec, fig_gene_spec, fig_overall
    
    return figures, corrs_overall



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
                   model_checkpoint,
                   checkpoint_name):
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
            ### write to logging file in saved model dir to model parameters and current epoch info    
        patience_counter = 0
        stop_criteria = False
        
    if (((current_epoch % save_freq) == 0) and (not stop_criteria)):
        print('Saving model...')

        
        model_checkpoint.save(checkpoint_name)
    
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
    parser.add_argument('--gcs_path',
                        dest='gcs_path',
                        help= 'google bucket containing preprocessed data')
    parser.add_argument('--gcs_path_TSS',
                        dest='gcs_path_TSS',
                        help= 'google bucket containing preprocessed data')
    parser.add_argument('--num_parallel', dest = 'num_parallel',
                        type=int, default=tf.data.AUTOTUNE,
                        help='thread count for tensorflow record loading')
    parser.add_argument('--batch_size', dest = 'batch_size',
                        default=1,
                        type=int, help='batch_size')
    parser.add_argument('--num_epochs', dest = 'num_epochs',
                        type=int, help='num_epochs')
    parser.add_argument('--num_examples_dict', dest = 'num_examples_dict',
                        default="human:34201,2213",
                        type=str, help='num_examples_dict')
    parser.add_argument('--val_examples_TSS', dest = 'val_examples_TSS',
                        type=int, help='val_examples_TSS')
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
    parser.add_argument('--lr_base',
                        dest='lr_base',
                        default="1.0e-03",
                        help='lr_base')
    parser.add_argument('--warmup_frac', 
                        dest = 'warmup_frac',
                        default=0.01,
                        type=float, help='warmup_frac')
    parser.add_argument('--filter_list',
                        dest='filter_list',
                        default="768,896,1024,1152,1280,1536",
                        help='filter_list')
    parser.add_argument('--heads_channels',
                        dest='heads_channels',
                        default="human:5313;mouse:1643",
                        help='heads_channels')
    parser.add_argument('--epsilon',
                        dest='epsilon',
                        default=1.0e-8,
                        type=float,
                        help= 'epsilon')
    parser.add_argument('--gradient_clip',
                        dest='gradient_clip',
                        type=str,
                        default="0.2",
                        help= 'gradient_clip')
    parser.add_argument('--savefreq',
                        dest='savefreq',
                        type=int,
                        help= 'savefreq')
    parser.add_argument('--enformer_checkpoint_path',
                        dest='enformer_checkpoint_path',
                        type=str,
                        default="/home/jupyter/dev/BE_CD69_paper_2022/enformer_fine_tuning/checkpoint/sonnet_weights",
                        help= 'enformer_checkpoint_path')
    parser.add_argument('--checkpoint_path',
                        dest='checkpoint_path',
                        type=str,
                        default=None,
                        help= 'checkpoint_path')
    parser.add_argument('--use_enformer_weights',
                        dest='use_enformer_weights',
                        type=str,
                        default="False",
                        help= 'use_enformer_weights')
    parser.add_argument('--train_model',
                        dest='train_model',
                        type=str,
                        default="False",
                        help= 'train_model')
    
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





def log2(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator