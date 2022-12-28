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
import pandas as pd
from datetime import datetime
import random

#import logging
#from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision

## custom modules

import metrics as metrics
import optimizers as optimizers
import schedulers as schedulers
import utils as utils

import training_utils_enformer_performer as training_utils
import seaborn as sns
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr  
from scipy import stats

def parse_bool_str(input_str):
    if input_str == 'False':
        return False
    else:
        return True
    
def parse_dict_input(input_str):
    out_dict={}
    if ';' in input_str:
        dict_items = input_str.split(';')
        for item in dict_items:
            out_dict[item.split(':')[0]]=int(item.split(':')[1])
    else:
        out_dict[input_str.split(':')[0]]=int(input_str.split(':')[1])
    
    return out_dict

def parse_dict_input_tuple(input_str,global_batch_size):
    out_dict={}
    if ';' in input_str:
        dict_items = input_str.split(';')
        for item in dict_items:
            out_dict[item.split(':')[0]]=(int(item.split(':')[1].split(',')[0]) // global_batch_size),\
                                            (int(item.split(':')[1].split(',')[1]) // global_batch_size)
    else:
        out_dict[input_str.split(':')[0]]=(int(input_str.split(':')[1].split(',')[0]) // global_batch_size),\
                                            (int(input_str.split(':')[1].split(',')[1]) // global_batch_size)
    
    return out_dict
    


 ## reformat 
# ===========================================================================#

def main():
    # ============== arg parse ==============================================# 
    parser = argparse.ArgumentParser(
        description='process input for enformer_performer training loop')
    parser = training_utils.parse_args(parser)
    args = parser.parse_args()

    # ============== define sweep options ==================== #
    sweep_config = {
            "name" : args.wandb_sweep_name,
            'method': "grid",
            'metric': {
                'name': 'hg_val_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'input_length': {
                    'values': [args.input_length]
                },
                'output_length': {
                    'values': [args.output_length]
                },
                'dropout_rate': {
                    'values': [float(x) for x in args.dropout_rate.split(',')]
                },
                'attention_dropout_rate': {
                    'values': [float(x) for x in args.attention_dropout_rate.split(',')]
                },
                'lr_base1': {
                    'values':[float(x) for x in args.lr_base1.split(',')]
                },
                'lr_base2': {
                    'values':[float(x) for x in args.lr_base2.split(',')]
                },
                'gradient_clip': {
                    'values': [float(x) for x in args.gradient_clip.split(',')]
                },
                'num_transformer_layers':{
                    'values': [int(x) for x in args.num_transformer_layers.split(',')]
                },
                'num_heads':{
                    'values': [int(x) for x in args.num_heads.split(',')]
                },
                'nb_random_features': {
                    'values':[int(x) for x in args.nb_random_features.split(',')]
                },
                'kernel_transformation': {
                    'values':[args.kernel_transformation]
                },
                'epsilon': {
                    'values':[args.epsilon]
                },
                'use_rot_emb': {
                    'values':[parse_bool_str(x) for x in args.use_rot_emb.split(',')]
                },
                'use_mask_pos': {
                    'values':[parse_bool_str(x) for x in args.use_mask_pos.split(',')]
                },
                'load_init': {
                    'values':[parse_bool_str(x) for x in args.load_init.split(',')]
                },
                'freeze_conv_layers': {
                    'values':[parse_bool_str(x) for x in args.freeze_conv_layers.split(',')]
                },
                'normalize': {
                    'values':[parse_bool_str(x) for x in args.normalize.split(',')]
                },
                'norm': {
                    'values':[parse_bool_str(x) for x in args.norm.split(',')]
                },
                'stable_variant': {
                    'values':[parse_bool_str(x) for x in args.stable_variant.split(',')]
                },
                'filter_list': {
                    'values': [[int(x) for x in args.filter_list.split(',')]]
                },
                'loss_type': {
                    'values':[str(x) for x in args.loss_type.split(',')]
                },
                'heads_channels': {
                    'values':[parse_dict_input(args.heads_channels)]
                },
                'model_type': {
                    'values':[args.model_type]
                }
            }
    }

    
    def sweep_train(config_defaults=None):
        # Set default values
        # Specify the other hyperparameters to the configuration, if any

        ## tpu initialization
        strategy = training_utils.tf_tpu_initialize(args.tpu_name,args.tpu_zone)
        mixed_precision.set_global_policy('mixed_bfloat16')
        
        
        g = tf.random.Generator.from_non_deterministic_state()
        ## rest must be w/in strategy scope
        with strategy.scope():
            config_defaults = {
                "lr_base": 0.01 ### will be overwritten
            }
            
            ### log training parameters
            wandb.init(config=config_defaults, 
                       project= args.wandb_project, 
                       entity=args.wandb_user)
            wandb.Table.MAX_ROWS = 2000000
            #wandb.init(mode="disabled")
            wandb.config.tpu=args.tpu_name
            wandb.config.gcs_path=args.gcs_path
            wandb.config.gcs_path_TSS=args.gcs_path_TSS
            wandb.config.num_epochs=args.num_epochs

            wandb.config.batch_size=args.batch_size
            wandb.config.warmup_frac=args.warmup_frac
            wandb.config.patience=args.patience
            wandb.config.min_delta=args.min_delta
            wandb.config.model_save_dir=args.model_save_dir
            wandb.config.model_save_basename=args.model_save_basename
            wandb.config.max_shift=args.max_shift
            
            if wandb.config.load_init:
                inits=training_utils.get_initializers(args.enformer_checkpoint_path)
                wandb.config.update({"filter_list": [768, 896, 1024, 1152, 1280, 1536]},
                                    allow_val_change=True)
            else:
                inits=None
                
            print(wandb.config.filter_list)
            
            run_name = '_'.join(['input_length-' + str(wandb.config.input_length),
                                 'load_init-' + str(wandb.config.load_init),
                                 'freeze-' + str(wandb.config.freeze_conv_layers),
                                 'LR1-' + str(wandb.config.lr_base1),
                                 'LR2-' + str(wandb.config.lr_base2),
                                 'T-' + str(wandb.config.num_transformer_layers),
                                 'F-' + str(wandb.config.filter_list[-1]),
                                 'D-' + str(wandb.config.dropout_rate),
                                 'K-' + str(wandb.config.kernel_transformation),
                                 'AD-' + str(wandb.config.attention_dropout_rate),
                                 'model-' + str(wandb.config.model_type)])
            wandb.run.name = run_name
            base_name = wandb.config.model_save_basename + "_" + run_name

            '''
            TPU init options
            '''

            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy=\
                tf.data.experimental.AutoShardPolicy.OFF
            options.deterministic=False
            options.experimental_threading.max_intra_op_parallelism=1
            tf.config.optimizer.set_jit(True)
            #options.experimental_slack = True

            NUM_REPLICAS = strategy.num_replicas_in_sync
            BATCH_SIZE_PER_REPLICA=wandb.config.batch_size
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA*NUM_REPLICAS
            print('global batch size:', GLOBAL_BATCH_SIZE)
            
            
            organism_dict = parse_dict_input_tuple(args.num_examples_dict,
                                                   GLOBAL_BATCH_SIZE)

            wandb.config.update({"total_steps" : (organism_dict['human'][0]) // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            wandb.config.update({"val_steps_TSS": args.val_examples_TSS // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            
            
            iterators = {}
            for key,val in wandb.config.heads_channels.items():
                iterators[key]=(wandb.config.gcs_path,val)

            tr_data_it_dict,val_data_it_dict,val_data_TSS_it= \
                    training_utils.return_distributed_iterators(iterators,
                                                                wandb.config.gcs_path_TSS,
                                                                GLOBAL_BATCH_SIZE,
                                                                wandb.config.input_length,
                                                                wandb.config.max_shift,
                                                                args.num_parallel,
                                                                args.num_epochs,
                                                                strategy,
                                                                options,
                                                                g)


            print('created dataset iterators')

            if wandb.config.model_type == 'enformer_performer':
                import enformer_performer
                print('loaded enformer_performer')
                model = enformer_performer.enformer_performer(num_transformer_layers=wandb.config.num_transformer_layers,
                                                              num_heads=wandb.config.num_heads,
                                                              heads_channels=wandb.config.heads_channels,
                                                              filter_list=wandb.config.filter_list,
                                                              dim=wandb.config.filter_list[-1] // wandb.config.num_heads,
                                                              d_model=wandb.config.filter_list[-1],
                                                              norm=wandb.config.norm,
                                                              max_seq_length=wandb.config.output_length,
                                                              nb_random_features=wandb.config.nb_random_features,
                                                              hidden_size=wandb.config.filter_list[-1],
                                                              numerical_stabilizer=0.001,
                                                              inits=inits,
                                                              stable_variant=wandb.config.stable_variant,
                                                              dropout_rate=wandb.config.dropout_rate,
                                                              attention_dropout_rate=wandb.config.attention_dropout_rate,
                                                              rel_pos_bins=wandb.config.output_length,
                                                              use_mask_pos=wandb.config.use_mask_pos,
                                                              use_rot_emb=wandb.config.use_rot_emb,
                                                              load_init=wandb.config.load_init,
                                                              freeze_conv_layers=wandb.config.freeze_conv_layers,
                                                              kernel_transformation=wandb.config.kernel_transformation,
                                                              normalize=wandb.config.normalize)
            elif wandb.config.model_type == 'enformer_performer_LN_only':
                import enformer_performer_LN_only
                print('loaded enformer_performer,LN only')
                model =\
                    enformer_performer_LN_only.enformer_performer_LN_only(num_transformer_layers=wandb.config.num_transformer_layers,
                                                              num_heads=wandb.config.num_heads,
                                                              heads_channels=wandb.config.heads_channels,
                                                              filter_list=wandb.config.filter_list,
                                                              dim=wandb.config.filter_list[-1] // wandb.config.num_heads,
                                                              d_model=wandb.config.filter_list[-1],
                                                              norm=wandb.config.norm,
                                                              max_seq_length=wandb.config.output_length,
                                                              nb_random_features=wandb.config.nb_random_features,
                                                              hidden_size=wandb.config.filter_list[-1],
                                                              numerical_stabilizer=0.001,
                                                              inits=inits,
                                                              dropout_rate=wandb.config.dropout_rate,
                                                              attention_dropout_rate=wandb.config.attention_dropout_rate,
                                                              rel_pos_bins=wandb.config.output_length,
                                                              use_mask_pos=wandb.config.use_mask_pos,
                                                              use_rot_emb=wandb.config.use_rot_emb,
                                                              load_init=wandb.config.load_init,
                                                              freeze_conv_layers=wandb.config.freeze_conv_layers,
                                                              kernel_transformation=wandb.config.kernel_transformation,
                                                              normalize=wandb.config.normalize)
            else:
                raise ValueError('model type not implemented')
            
            checkpoint_name = wandb.config.model_save_dir + "/" + \
                            wandb.config.model_save_basename + "_" + wandb.run.name


            model_checkpoint = tf.train.Checkpoint(module=model)
                                                          
            print('initialized model')
            scheduler1= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base1,
                decay_steps=wandb.config.total_steps*wandb.config.num_epochs, alpha=1.0)
            scheduler1=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base1,
                                         warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps,
                                         decay_schedule_fn=scheduler1)
            
            optimizer1 = tf.keras.optimizers.Adam(learning_rate=scheduler1,
                                                  epsilon=wandb.config.epsilon)
            #####
            scheduler2= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base2,
                decay_steps=wandb.config.total_steps*wandb.config.num_epochs, alpha=1.0)
            scheduler2=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base2,
                                         warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps,
                                         decay_schedule_fn=scheduler2)
            
            optimizer2 = tf.keras.optimizers.Adam(learning_rate=scheduler2,
                                              epsilon=wandb.config.epsilon)
            #####
            optimizers_in = optimizer1,optimizer2
            

            metric_dict = {}

            dist_train_step,val_step_h,val_step_m,val_step_TSS,build_step, metric_dict = \
                            training_utils.return_train_val_functions(model,
                                                                      organism_dict['human'][0],
                                                                      organism_dict['human'][1],
                                                                      organism_dict['mouse'][1],
                                                                      wandb.config.val_steps_TSS,
                                                                      optimizers_in,
                                                                      strategy,
                                                                      metric_dict,
                                                                      GLOBAL_BATCH_SIZE,
                                                                      wandb.config.gradient_clip,
                                                                      BATCH_SIZE_PER_REPLICA,
                                                                      loss_fn_main='poisson')
            


            print('finished loading training/val loop functions')
            global_step = 0
            val_losses = []
            val_pearsons = []
            val_R2 = []
            patience_counter = 0
            stop_criteria = False
            best_epoch = 0
            for epoch_i in range(1, wandb.config.num_epochs+1):
                if epoch_i == 1:
                    print('building model')
                    build_step(val_data_it_dict['human'])
                    total_params = 0
                    for k in model.trainable_variables:
                        var = k.values[0]
                        total_params += tf.size(var)
                    print('total params: ' + str(total_params)) 
                
                print('starting epoch_', str(epoch_i))
                start = time.time()
                
                dist_train_step(tr_data_it_dict['human'],
                                tr_data_it_dict['mouse'])
                #for organism in organism_dict.keys():
                    
                    #train_step_dict[organism](tr_data_it_dict[organism])
                wandb.log({'human_train_loss': metric_dict['human_tr'].result().numpy()},
                          step=epoch_i)
                wandb.log({'mouse_train_loss': metric_dict['mouse_tr'].result().numpy()},
                          step=epoch_i)

                end = time.time()
                duration = (end - start) / 60.
                print('completed epoch ' + str(epoch_i))
                print('human_train_loss: ' + str(metric_dict['human_tr'].result().numpy()))

                print('training duration(mins): ' + str(duration))
                
                start = time.time()
                val_step_h(val_data_it_dict['human'])
                val_step_m(val_data_it_dict['mouse'])
                
                for organism in organism_dict.keys():
                    wandb.log({organism + '_val_loss': metric_dict[organism + '_val'].result().numpy()},
                              step=epoch_i)
                    pearsonsR=metric_dict[organism+'_pearsonsR'].result()['PearsonR'].numpy()
                    
                    wandb.log({organism + '_all_tracks_pearsons': np.nanmean(pearsonsR),
                               organism+'_DNASE_pearsons': np.nanmean(pearsonsR[:684]),
                               organism+'_CHIP_pearsons': np.nanmean(pearsonsR[684:4675]),
                               organism+'_CAGE_pearsons': np.nanmean(pearsonsR[4675:])},
                              step=epoch_i)

                    R2=metric_dict[organism+'_R2'].result()['R2'].numpy()
                    wandb.log({organism + '_all_tracks_R2': np.nanmean(R2),
                               organism+'_DNASE_R2': np.nanmean(R2[:684]),
                               organism+'_CHIP_R2': np.nanmean(R2[684:4675]),
                               organism+'_CAGE_R2': np.nanmean(R2[4675:])},
                              step=epoch_i)
                
                print('human_val_loss: ' + str(metric_dict['human_val'].result().numpy()))
                val_losses.append(metric_dict['human_val'].result().numpy())
                
                val_step_TSS(val_data_TSS_it)
                
                print('computing TSS quant metrics')
                
                y_trues = metric_dict['hg_corr_stats'].result()['y_trues'].numpy()
                y_preds = metric_dict['hg_corr_stats'].result()['y_preds'].numpy()
                cell_types = metric_dict['hg_corr_stats'].result()['cell_types'].numpy()
                gene_map = metric_dict['hg_corr_stats'].result()['gene_map'].numpy()

                figures,corrs_overall= training_utils.make_plots(y_trues,
                                                                 y_preds,
                                                                 cell_types,
                                                                 gene_map)

                print('returned TSS centered correlations and figures')
                fig_cell_spec, fig_gene_spec, fig_overall=figures 

                cell_spec_mean_corrs, \
                    gene_spec_mean_corrs = corrs_overall
                
                val_pearsons.append(cell_spec_mean_corrs)
                
                print('hg_RNA_pearson: ' + str(cell_spec_mean_corrs))
                
                wandb.log({'gene_spec_mean_corrs': gene_spec_mean_corrs,
                           'cell_spec_mean_corrs': cell_spec_mean_corrs},
                          step=epoch_i)
                try:
                    wandb.log({'hg_OVERALL_TSS_predictions': fig_overall,
                               'cross_dataset_dist': fig_cell_spec,
                               'cross_gene_dist': fig_gene_spec},
                              step=epoch_i)
                except IndexError:
                    pass
                
                
                if wandb.config.model_type == 'enformer_performer':
                    fig_gamma,fig_beta,fig_moving_means,fig_moving_vars=\
                            training_utils.extract_batch_norm_stats(model)
                    wandb.log({'gamma': fig_gamma,
                               'beta': fig_beta,
                               'moving_mean': fig_moving_means,
                               'moving_var': fig_moving_vars},
                              step=epoch_i)
                
                end = time.time()
                duration = (end - start) / 60.
                print('completed epoch ' + str(epoch_i) + ' validation')
                print('validation duration(mins): ' + str(duration))
                
                if (epoch_i > 2):
                    stop_criteria,patience_counter,best_epoch = \
                        training_utils.early_stopping(current_val_loss=val_losses[-1],
                                                        logged_val_losses=val_losses,
                                                        current_pearsons=val_pearsons[-1],
                                                        logged_pearsons=val_pearsons,
                                                        current_epoch=epoch_i,
                                                        best_epoch=best_epoch,
                                                        save_freq=args.savefreq,
                                                        patience=wandb.config.patience,
                                                        patience_counter=patience_counter,
                                                        min_delta=wandb.config.min_delta,
                                                        model=model,
                                                        save_directory=wandb.config.model_save_dir,
                                                        saved_model_basename=base_name)
                    #plt.close('all')
                    print('patience counter at: ' + str(patience_counter))
                for key, item in metric_dict.items():
                    item.reset_state()
                if stop_criteria:
                    print('early stopping at: epoch ' + str(epoch_i))
                    break
                    
            print('saving model at: epoch ' + str(epoch_i))
            print('best model was at: epoch ' + str(best_epoch))
            model.save_weights(wandb.config.model_save_dir + "/" + base_name + "/final/saved_model")

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=sweep_train)
    #sweep_train()

##########################################################################
if __name__ == '__main__':
    main()

