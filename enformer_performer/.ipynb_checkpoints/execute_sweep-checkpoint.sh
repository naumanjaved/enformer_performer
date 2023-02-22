#!/bin/bash -l

python3 train_model.py \
            --tpu_name="po1" \
            --tpu_zone="us-east1-d" \
            --wandb_project="enformer_performer" \
            --wandb_user="njaved" \
            --wandb_sweep_name="enformer_performer" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://genformer_data/expanded_originals/196k" \
            --gcs_path_TSS="gs://genformer_data/expanded_originals/196k/human/tfrecords_tss" \
            --num_epochs=100 \
            --warmup_frac=0.01 \
            --patience=50\
            --min_delta=0.00001 \
            --model_save_dir="gs://picard-testing-176520/enformer_performer/models" \
            --model_save_basename="enformer_performer_230221" \
            --lr_base1="1.0e-04" \
            --lr_base2="1.0e-04" \
            --wd_1="0.0" \
            --wd_2="0.0" \
            --decay_frac="0.75" \
            --gradient_clip="5.0" \
            --BN_momentum="0.90" \
            --epsilon=1.0e-8 \
            --num_parallel=8 \
            --dropout_rate=0.40 \
            --attention_dropout_rate=0.05 \
            --savefreq=5 \
            --input_length=196608 \
            --output_length=1536 \
            --val_examples_TSS=2134 \
            --load_init="False" \
            --freeze_conv_layers="False" \
            --num_examples_dict="human:34021,2213;mouse:29295,2209" \
            --num_transformer_layers=8 \
            --num_heads=8 \
            --stable_variant="True" \
            --optimizer="adamw" \
            --heads_channels="human:5313;mouse:1643" \
            --kernel_transformation="relu_kernel_transformation" \
            --block_type="enformer" \
            --use_max_pool="False" 
            
