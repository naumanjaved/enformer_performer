#!/bin/bash -l

python3 train_model_batchnorm_experiments.py \
            --tpu_name="pod1" \
            --tpu_zone="us-east1-d" \
            --wandb_project="enformer_performer_batchnorm" \
            --wandb_user="njaved" \
            --wandb_sweep_name="enformer_performer_batchnorm" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://genformer_data/expanded_originals/196k" \
            --gcs_path_TSS="gs://genformer_data/expanded_originals/196k/human/tfrecords_tss" \
            --num_epochs=40 \
            --warmup_frac=1.50 \
            --patience=25\
            --min_delta=0.00001 \
            --model_save_dir="gs://picard-testing-176520/enformer_batchnorms/models" \
            --model_save_basename="enformer_batchnorms" \
            --lr_base1="1.0e-06" \
            --lr_base2="8.0e-05" \
            --gradient_clip="1.0" \
            --epsilon=1.0e-8 \
            --num_parallel=8 \
            --savefreq=45 \
            --val_examples_TSS=2134 \
            --load_init="True" \
            --freeze_conv_layers="False" \
            --enformer_checkpoint_path="sonnet_weights" \
            --num_examples_dict="human:34021,2213;mouse:29295,2209" \
            --model_type='enformer_performer' \
            --num_transformer_layers=11 \
            --num_heads=8 \
            --stable_variant="True,False" \
            --heads_channels="human:5313;mouse:1643" \
            --kernel_transformation="relu_kernel_transformation"
            
