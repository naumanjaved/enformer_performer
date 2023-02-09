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
import time
from datetime import datetime
import random

import seaborn as sns

import logging
from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
#os.environ['TPU_LOAD_LIBRARY']='0'
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
import tensorflow as tf


import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision
from scipy.stats.stats import pearsonr  
from scipy.stats.stats import spearmanr  
## custom modules
import metrics as metrics
from optimizers import *
import schedulers as schedulers

from scipy import stats

import enformer_vanilla as enformer

import pandas as pd
from scipy.stats.stats import pearsonr, spearmanr
from scipy.stats import zscore

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='node-4')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy=\
        tf.data.experimental.AutoShardPolicy.OFF
    options.deterministic=False
    options.experimental_threading.max_intra_op_parallelism=1
    #mixed_precision.set_global_policy('mixed_bfloat16')
    #options.num_devices = 64

    BATCH_SIZE_PER_REPLICA = 1 # batch size 24, use LR ~ 2.5 e -04
    NUM_REPLICAS = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_REPLICAS
    
    
def deserialize_val_TSS(serialized_example,input_length,max_shift, out_length,num_targets):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
        'tss_mask': tf.io.FixedLenFeature([], tf.string),
        'gene_name': tf.io.FixedLenFeature([], tf.string)
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
                        (out_length, num_targets))
    
    tss_mask = tf.io.parse_tensor(data['tss_mask'],
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

with strategy.scope():
    
    list_files_val = (tf.io.gfile.glob(os.path.join("gs://genformer_data/expanded_originals/196k/human/tfrecords_tss",
                                                "tssmask-valid*.tfr")))
    
    files = tf.data.Dataset.list_files(list_files_val)
    dataset_build = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      num_parallel_reads=4)
    dataset_build = dataset_build.with_options(options)
    dataset_build = dataset_build.map(lambda record: deserialize_val_TSS(record,
                                                     196608,
                                                     10,
                                                     896,
                                                     5313),
                          deterministic=False,
                          num_parallel_calls=4)
    

    dataset_build=dataset_build.repeat(1).batch(1).prefetch(1)
    val_dist_build= strategy.experimental_distribute_dataset(dataset_build)
    val_dist_build_it = iter(val_dist_build)
    
    model = enformer.Enformer()
    
    def run_build_dist(iterator):
        @tf.function
        def run_build(inputs):
            sequence = inputs['sequence']
            model(sequence,is_training=False)
        strategy.run(run_build,
                     args=(next(iterator),))
    
    run_build_dist(val_dist_build_it)
    ### build model then load weights
    
    checkpoint_options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    checkpoint = tf.train.Checkpoint(module=model)#,options=options)
    tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    latest = tf.train.latest_checkpoint("/home/jupyter/dev/BE_CD69_paper_2022/enformer_fine_tuning/checkpoint/sonnet_weights/")

    checkpoint.restore(latest,options=checkpoint_options).assert_consumed()
    
    
    
with strategy.scope():
    overall_step = 0
    pearsons_list=[]
    r2_list=[]

    #for k,file in enumerate(list_files_val):
    pred_gene_cents_all = []
    true_gene_cents_all = []
    gene_names_all = []
    cell_types_all = []

    list_files_val = (tf.io.gfile.glob(os.path.join("gs://genformer_data/expanded_originals/196k/human/tfrecords_tss",
                                                "tssmask-valid*.tfr")))
    
    files = tf.data.Dataset.list_files(list_files_val)

    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      num_parallel_reads=4)
    dataset = dataset.with_options(options)
    dataset = dataset.map(lambda record: deserialize_val_TSS(record,
                                                     196608,
                                                     10,
                                                     896,
                                                     5313),
                          deterministic=False,
                          num_parallel_calls=4)

    dataset=dataset.repeat(2).batch(8,drop_remainder=True).prefetch(1)
    val_dist= strategy.experimental_distribute_dataset(dataset)
    val_dist_it = iter(val_dist)

    hg_corr_stats = metrics.correlation_stats_gene_centered(name='hg_corr_stats')

    def dist_run_model(iterator):
        
        @tf.function
        def run_model(inputs):
            sequence = inputs['sequence']
            target = inputs['target'][:,:,4675:]

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

        for _ in range(277):
            pred_rep, true_rep, gene_rep, cell_type_rep = strategy.run(run_model,
                                                                       args=(next(iterator),))
            pred_reshape = tf.reshape(strategy.gather(pred_rep, axis=0), [-1]) # reshape to 1D
            true_reshape = tf.reshape(strategy.gather(true_rep, axis=0), [-1])
            cell_type_reshape = tf.reshape(strategy.gather(cell_type_rep, axis=0), [-1])
            gene_map_reshape = tf.reshape(strategy.gather(gene_rep, axis=0), [-1])

            ta_pred = ta_pred.write(_, pred_reshape)
            ta_true = ta_true.write(_, true_reshape)
            ta_celltype = ta_celltype.write(_, cell_type_reshape)
            ta_genemap = ta_genemap.write(_, gene_map_reshape)
            
            
        hg_corr_stats.update_state(ta_true.concat(),
                                  ta_pred.concat(),
                                  ta_celltype.concat(),
                                  ta_genemap.concat())
        ta_true.close()
        ta_pred.close()
        ta_celltype.close()
        ta_genemap.close()
        
    dist_run_model(val_dist_it)
    
    results_df = pd.DataFrame()
    results_df['true'] = hg_corr_stats.result()['y_trues'].numpy()
    results_df['pred'] = hg_corr_stats.result()['y_preds'].numpy()
    results_df['gene_encoding']  = hg_corr_stats.result()['gene_map'].numpy()
    results_df['cell_type_encoding'] = hg_corr_stats.result()['cell_types'].numpy()
    

    results_df=results_df.groupby(['gene_encoding', 'cell_type_encoding']).agg({'true': 'sum', 'pred': 'sum'})
    results_df['true'] = np.log2(1.0+results_df['true'])
    results_df['pred'] = np.log2(1.0+results_df['pred'])

    results_df['true_zscore']=results_df.groupby(['cell_type_encoding']).true.transform(lambda x : zscore(x))
    results_df['pred_zscore']=results_df.groupby(['cell_type_encoding']).pred.transform(lambda x : zscore(x))

    true_zscore=results_df[['true_zscore']].to_numpy()[:,0]

    pred_zscore=results_df[['pred_zscore']].to_numpy()[:,0]

    cell_specific_corrs=results_df.groupby('cell_type_encoding')[['true_zscore','pred_zscore']].corr(method='pearson').unstack().iloc[:,1].tolist()

    results_df.to_csv('gene_level_predictions.tsv',sep='\t',index=False,header=True)

    cell_specific_corrs_arr = np.array(cell_specific_corrs)

    np.savetxt('cell_specific_corrs_arr.out', cell_specific_corrs_arr, delimiter=',')
        