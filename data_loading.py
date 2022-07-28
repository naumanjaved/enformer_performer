import tensorflow as tf
import sonnet as snt
#from tqdm import tqdm
import tensorflow.experimental.numpy as tnp
import pandas as pd
import time
import os
import sys

import glob
import json
import functools

'''
Data loading functionality from Enformer paper
'''


# @title `get_targets(organism)`
def get_targets(organism):
    targets_txt = f'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_{organism}.txt'
    return pd.read_csv(targets_txt, sep='\t')

def organism_path(organism, GCS_data_loc):
    return os.path.join(GCS_data_loc, organism)


def get_dataset(organism, 
                split,
                GCS_data_loc,
                indices,
                batch_size,
                num_parallel, 
                repeat, 
                prefetch):
    metadata = get_metadata(organism, GCS_data_loc)
    dataset = tf.data.TFRecordDataset(tfrecord_files(organism, GCS_data_loc, split),
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_parallel)
    dataset = dataset.map(functools.partial(deserialize, 
                                            metadata=metadata,
                                            indices=indices),
                          num_parallel_calls=num_parallel)
    #print(dataset)
    return dataset.batch(batch_size).repeat().prefetch(prefetch)


def get_metadata(organism, GCS_data_loc):
    path = os.path.join(organism_path(organism, 
                                      GCS_data_loc), 
                        'statistics.json')
    with tf.io.gfile.GFile(path, 'r') as f:
        return json.load(f)


def tfrecord_files(organism, GCS_data_loc, subset):
    # Sort the values by int(*).
    return sorted(tf.io.gfile.glob(os.path.join(
        organism_path(organism, GCS_data_loc), 'tfrecords', f'{subset}-*.tfr')), 
                  key=lambda x: int(x.split('-')[-1].split('.')[0]))


def deserialize(serialized_example, metadata, indices):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (metadata['target_length'], metadata['num_targets']))
    target = tf.gather(tf.cast(target, tf.float32),
                       batch_dims=0,
                       axis=1,
                       indices=indices)

    return {'sequence': sequence,
            'target': target}
