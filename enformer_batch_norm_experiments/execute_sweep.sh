#!/bin/bash -l

python3 train_model_batchnorm_experiments.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="enformer_performer" \
            --wandb_user="njaved" \
            --wandb_sweep_name="enformer_performer" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://genformer_data/expanded_originals/196k" \
            --gcs_path_TSS="gs://genformer_data/expanded_originals/196k/human/tfrecords_tss" \
            --num_epochs=40 \
            --warmup_frac=1.50 \
            --patience=25\
            --min_delta=0.00001 \
            --model_save_dir="gs://picard-testing-176520/enformer_performer/models" \
            --model_save_basename="enformer_performer" \
            --lr_base1="1.0e-06" \
            --lr_base2="8.0e-05" \
            --wd_1="1.0e-06" \
            --wd_2="8.0e-05" \
            --decay_frac="0.80" \
            --gradient_clip="1.0" \
            --BN_momentum="0.80" \
            --post_BN_dropout_rate="0.20" \
            --epsilon=1.0e-8 \
            --num_parallel=8 \
            --savefreq=45 \
            --val_examples_TSS=2134 \
            --load_init="False" \
            --freeze_conv_layers="False" \
            --num_examples_dict="human:34021,2213;mouse:29295,2209" \
            --num_transformer_layers=11 \
            --num_heads=8 \
            --stable_variant="True,False" \
            --heads_channels="human:5313;mouse:1643" \
            --kernel_transformation="relu_kernel_transformation"
            
