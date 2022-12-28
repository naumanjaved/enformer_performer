from typing import Any, Callable, Dict, Optional, Text, Union, Iterable

import tensorflow.experimental.numpy as tnp
import tensorflow as tf

from tensorflow.keras import layers as kl

import tensorflow_addons as tfa
from tensorflow.keras import regularizers

from tensorflow.keras.layers.experimental import SyncBatchNormalization as syncbatchnorm
from layers import *

SEQUENCE_LENGTH=196608
TARGET_LENGTH=896

@tf.keras.utils.register_keras_serializable()
class enformer_performer(tf.keras.Model):
    def __init__(self,
                 num_transformer_layers: int = 11,
                 num_heads: int = 8,
                 heads_channels: dict = {'human': 2696,
                                         'mouse': 987,
                                         'rat': 13,
                                         'canine': 13,
                                         'rhesus': 15},
                 filter_list=[768,896,1024,1152,1280,1536],
                 dim=192,
                 d_model=1536,
                 norm=True,
                 max_seq_length=1536,
                 nb_random_features=256,
                 hidden_size=1536,
                 numerical_stabilizer=0.001,
                 attention_dropout_rate=0.10,
                 dropout_rate=0.40,
                 rel_pos_bins=1536,
                 use_mask_pos=False,
                 use_rot_emb=True,
                 load_init=False,
                 freeze_conv_layers=False,
                 stable_variant=True,
                 inits=None,
                 kernel_transformation="softmax_kernel_transformation",
                 normalize=True,
                 seed=5,
                 name: str = 'enformer_performer',
                 **kwargs):
        """ 'enformer_performer' model based on Enformer for predicting RNA-seq from atac + sequence
        Args: to do 
        
        
          name: model name
        """

        super(enformer_performer, self).__init__(name=name,**kwargs)
        self.heads_channels = heads_channels
        
        self.dropout_rate=dropout_rate
        
        self.num_transformer_layers=num_transformer_layers
        self.num_heads=num_heads
        self.dim=dim
        self.d_model=d_model
        self.norm=norm
        self.max_seq_length=max_seq_length
        self.nb_random_features=nb_random_features
        self.hidden_size=hidden_size
        self.numerical_stabilizer=numerical_stabilizer
        self.attention_dropout_rate=attention_dropout_rate
        self.rel_pos_bins=rel_pos_bins
        self.use_rot_emb=use_rot_emb
        self.use_mask_pos=use_mask_pos
        self.kernel_transformation=kernel_transformation
        self.normalize=normalize
        self.seed=seed
        self.inits=inits
        self.load_init=load_init
        self.freeze_conv_layers=freeze_conv_layers
        
        if self.load_init:
            self.filter_list= [768,896,1024,1152,1280,1536]
        else:
            self.filter_list=filter_list
        
        
        def enf_conv_block(filters, 
                           width=1, 
                           w_init='glorot_uniform', 
                           padding='same', 
                           name='conv_block',
                           beta_init=None,
                           gamma_init=None,
                           mean_init=None,
                           var_init=None,
                           kernel_init=None,
                           bias_init=None,
                           train=True,
                           **kwargs):
            return tf.keras.Sequential([
              syncbatchnorm(axis=-1,
                            center=True,
                            scale=True,
                            beta_initializer=beta_init if self.load_init else "zeros",
                            gamma_initializer=gamma_init if self.load_init else "ones",
                            trainable=train,
                            moving_mean_initializer=mean_init if self.load_init else "zeros",
                            moving_variance_initializer=var_init if self.load_init else "ones",
                            **kwargs),
              tfa.layers.GELU(),
              kl.Conv1D(filters,
                         width, 
                         kernel_initializer=kernel_init if self.load_init else w_init,
                         bias_initializer=bias_init if self.load_init else bias_init,
                         trainable=train,
                         padding=padding, **kwargs)
            ], name=name)
        
        ### conv stack for sequence inputs
        self.stem_conv = kl.Conv1D(filters= int(self.filter_list[-1]) // 2,
                                   kernel_size=15,
                                   kernel_initializer=self.inits['stem_conv_k'] if self.load_init else 'glorot_uniform',
                                   bias_initializer=self.inits['stem_conv_b'] if self.load_init else 'zeros',
                                   strides=1,
                                   trainable=False if self.freeze_conv_layers else True,
                                   padding='same')
                                   #data_format='channels_last')
        self.stem_res_conv=Residual(enf_conv_block(int(self.filter_list[-1]) // 2, 1,
                                                   beta_init=self.inits['stem_res_conv_BN_b'] if self.load_init else None,
                                                   gamma_init=self.inits['stem_res_conv_BN_g'] if self.load_init else None,
                                                   mean_init=self.inits['stem_res_conv_BN_m'] if self.load_init else None,
                                                   var_init=self.inits['stem_res_conv_BN_v'] if self.load_init else None,
                                                   kernel_init=self.inits['stem_res_conv_k'] if self.load_init else None,
                                                   bias_init=self.inits['stem_res_conv_b'] if self.load_init else None,
                                                   train=False if self.freeze_conv_layers else True,
                                                   name='pointwise_conv_block'))
        self.stem_pool = SoftmaxPooling1D(per_channel=True,
                                          w_init_scale=2.0,
                                          pool_size=2,
                                          k_init=self.inits['stem_pool'] if self.load_init else None,
                                          train=False if self.freeze_conv_layers else True,
                                          name ='stem_pool')

        
        self.conv_tower = tf.keras.Sequential([
            tf.keras.Sequential([
                enf_conv_block(num_filters, 
                               5, 
                               beta_init=self.inits['BN1_b_' + str(i)] if self.load_init else None,
                               gamma_init=self.inits['BN1_g_' + str(i)] if self.load_init else None,
                               mean_init=self.inits['BN1_b_' + str(i)] if self.load_init else None,
                               var_init=self.inits['BN1_v_' + str(i)] if self.load_init else None,
                               kernel_init=self.inits['conv1_k_' + str(i)] if self.load_init else None,
                               bias_init=self.inits['conv1_b_' + str(i)] if self.load_init else None,
                               train=False if self.freeze_conv_layers else True,
                               padding='same'),
                Residual(enf_conv_block(num_filters, 1, 
                                       beta_init=self.inits['BN2_b_' + str(i)] if self.load_init else None,
                                       gamma_init=self.inits['BN2_g_' + str(i)] if self.load_init else None,
                                       mean_init=self.inits['BN2_b_' + str(i)] if self.load_init else None,
                                       var_init=self.inits['BN2_v_' + str(i)] if self.load_init else None,
                                       kernel_init=self.inits['conv2_k_' + str(i)] if self.load_init else None,
                                       bias_init=self.inits['conv2_b_' + str(i)] if self.load_init else None,
                                        train=False if self.freeze_conv_layers else True,
                                        name='pointwise_conv_block')),
                SoftmaxPooling1D(per_channel=True,
                                 w_init_scale=2.0,
                                 k_init=self.inits['pool_'+str(i)] if self.load_init else None,
                                 train=False if self.freeze_conv_layers else True,
                                 pool_size=2),
                ],
                       name=f'conv_tower_block_{i}')
            for i, num_filters in enumerate(self.filter_list)], name='conv_tower')
        
        

        self.sin_pe = abs_sin_PE(name='sin_pe',
                                  **kwargs)
        if stable_variant:
            self.performer = Performer_Encoder_stable(num_layers=self.num_transformer_layers,
                                               num_heads=self.num_heads,
                                               dim = self.dim,
                                               d_model=self.d_model,
                                               norm=self.norm,
                                               max_seq_length=self.max_seq_length,
                                               nb_random_features=self.nb_random_features,
                                               hidden_size=self.hidden_size,
                                               numerical_stabilizer=self.numerical_stabilizer,
                                               attention_dropout=self.attention_dropout_rate,
                                               rel_pos_bins=self.rel_pos_bins,
                                               use_rot_emb=self.use_rot_emb,
                                               use_mask_pos=self.use_mask_pos,
                                               kernel_transformation=self.kernel_transformation,
                                               normalize=self.normalize,
                                               seed = self.seed,
                                               name = 'performer',
                                               **kwargs)
        else:
            self.performer = Performer_Encoder(num_layers=self.num_transformer_layers,
                                               num_heads=self.num_heads,
                                               dim = self.dim,
                                               d_model=self.d_model,
                                               norm=self.norm,
                                               max_seq_length=self.max_seq_length,
                                               nb_random_features=self.nb_random_features,
                                               hidden_size=self.hidden_size,
                                               numerical_stabilizer=self.numerical_stabilizer,
                                               attention_dropout=self.attention_dropout_rate,
                                               rel_pos_bins=self.rel_pos_bins,
                                               use_rot_emb=self.use_rot_emb,
                                               use_mask_pos=self.use_mask_pos,
                                               kernel_transformation=self.kernel_transformation,
                                               normalize=self.normalize,
                                               seed = self.seed,
                                               name = 'performer',
                                               **kwargs)


        self.crop_final = TargetLengthCrop1D(uncropped_length=1536, 
                                             target_length=TARGET_LENGTH,
                                             name='target_input')
        
        
        self.final_pointwise_conv = enf_conv_block(filters=self.filter_list[-1] * 2,
                                                  **kwargs,
                                                  name = 'final_pointwise_rna')
        self.dropout = kl.Dropout(rate=self.dropout_rate / 8,
                                  **kwargs)
        self.gelu = tfa.layers.GELU()



        self.heads = {
            head: kl.Dense(num_channels,
                           activation='softplus',
                           use_bias=True)  for head, num_channels in heads_channels.items()
        }
        
        
    def call(self, inputs, training:bool=True):
        
        x = self.stem_conv(inputs,
                           training=training)
        x = self.stem_res_conv(x,
                               training=training)
        x = self.stem_pool(x,
                           training=training)
        x = self.conv_tower(x,
                            training=training)
        x = self.sin_pe(x)
        x,att_matrices = self.performer(x,
                           training=training)
        x = self.crop_final(x)
        x = self.final_pointwise_conv(x,
                                      training=training)
        x = self.dropout(x,
                        training=training)
        x = self.gelu(x)

        out= {head: head_module(x) for head, head_module in self.heads.items()}
        return out
    

    def get_config(self):
        config = {
            "dropout_rate":self.dropout_rate,
            "heads_channels": self.heads_channels,
            "num_transformer_layers": self.num_transformer_layers,
            "numerical_stabilizer": self.numerical_stabilizer,
            "kernel_transformation": self.kernel_transformation,
            "nb_random_features": self.nb_random_features,
            "d_model":self.d_model,
            "norm":self.norm,
            "dim":self.dim,
            "max_seq_length":self.max_seq_length,
            "rel_pos_bins":self.rel_pos_bins,
            "use_rot_emb":self.use_rot_emb,
            "use_mask_pos":self.use_mask_pos,
            "normalize":self.normalize,
            "attention_dropout" : self.attention_dropout,
            "seed":self.seed
            
        }
        
        base_config = super().get_config()
        return {**base_config, **config}

        
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def predict_on_batch(self, inputs, training:bool=False):
        
        x = self.stem_conv(inputs,
                           training=training)
        x = self.stem_res_conv(x,
                               training=training)
        x = self.stem_pool(x,
                           training=training)
        x = self.conv_tower(x,
                            training=training)
        x = self.sin_pe(x)
        x,att_matrices = self.performer(x,
                           training=training)
        x = self.crop_final(x)
        x = self.final_pointwise_conv(x,
                                      training=training)
        x = self.dropout(x,
                        training=training)
        x = self.gelu(x)

        return {head: head_module(x) for head, head_module in self.heads.items()}, att_matrices

def exponential_linspace_int(start, end, num, divisible_by=1):
    """Exponentially increasing values of integers."""
    def _round(x):
        return tf.cast((tf.math.round(x / divisible_by) * divisible_by),dtype=tf.int32)

    base = tf.cast(tnp.exp(tnp.log(end / start) / (num - 1)), dtype=tf.float32)
    return [_round(start * base**i) for i in range(num)]
