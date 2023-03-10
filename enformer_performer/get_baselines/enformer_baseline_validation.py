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
from scipy import stats

import enformer_vanilla as enformer

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='node-1')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy=\
        tf.data.experimental.AutoShardPolicy.OFF
    options.deterministic=False
    options.experimental_threading.max_intra_op_parallelism=1
    mixed_precision.set_global_policy('mixed_bfloat16')
    #options.num_devices = 64

    BATCH_SIZE_PER_REPLICA = 1 # batch size 24, use LR ~ 2.5 e -04
    NUM_REPLICAS = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_REPLICAS
    


def deserialize_val(serialized_example,input_length,max_shift, out_length,num_targets):
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
                        (out_length, num_targets))
    
    
    return {'sequence': tf.ensure_shape(sequence,
                                        [input_length,4]),
            'target': tf.ensure_shape(target,
                                      [896,num_targets])}


pearsonsR = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
R2 = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})


with strategy.scope():
    
    list_files_val = (tf.io.gfile.glob(os.path.join("gs://genformer_data/expanded_originals/196k",
                                                "human",
                                                "tfrecords",
                                                "valid*.tfr")))
    
    files = tf.data.Dataset.list_files(list_files_val)
    dataset_build = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      num_parallel_reads=4)
    dataset_build = dataset_build.with_options(options)
    dataset_build = dataset_build.map(lambda record: deserialize_val(record,
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
    latest = tf.train.latest_checkpoint("gs://picard-testing-176520/sonnet_weights/sonnet_weights")

    checkpoint.restore(latest,options=checkpoint_options).assert_consumed()
    
    overall_step = 0
    pearsons_list=[]
    r2_list=[]
    def dist_run_model(iterator):
        @tf.function
        def run_model(inputs):
            sequence = inputs['sequence']
            target = inputs['target']
            return model(sequence,is_training=False)['human'],target
        
        dist_out,dist_target = strategy.run(run_model,
                                args=(next(iterator),))
        return dist_out,dist_target
    
    for file in list_files_val:

        dataset = tf.data.TFRecordDataset(file,
                                          compression_type='ZLIB',
                                          num_parallel_reads=4)
        dataset = dataset.with_options(options)
        dataset = dataset.map(lambda record: deserialize_val(record,
                                                         196608,
                                                         10,
                                                         896,
                                                         5313),
                              deterministic=False,
                              num_parallel_calls=4)

        dataset=dataset.repeat(2).batch(8).prefetch(1)
        val_dist= strategy.experimental_distribute_dataset(dataset)
        val_dist_it = iter(val_dist)

        for step in range(32):
            try:
                dist_out,dist_target = dist_run_model(val_dist_it)
                for i in range(8):
                    dist_out_vals = dist_out.values[i]
                    dist_target_vals = dist_target.values[i]
                    pearsonsR.update_state(dist_out_vals,
                                           dist_target_vals)
                    R2.update_state(dist_out_vals,
                                    dist_target_vals)


                    overall_step += 1
                    
            except StopIteration:
                print('reached end')
                break
        print('overall step:' + str(overall_step))
                
                
    pearsonsR_np = pearsonsR.result()['PearsonR'].numpy()
    r2_np = R2.result()['R2'].numpy()
    
    np.savetxt('pearsonsR_array_enformer_validation.out', pearsonsR_np, delimiter=',')
    np.savetxt('r2_array_enformer_validation.out', r2_np, delimiter=',')
    
