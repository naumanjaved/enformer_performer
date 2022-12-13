#!/bin/bash -l

python3 train_model_batchnorm_experiments.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="enformer_performer_batchnorm" \
            --wandb_user="njaved" \
            --wandb_sweep_name="enformer_performer_batchnorm" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://genformer_data/data_noTF" \
            --gcs_path_TSS="gs://genformer_data/data_noTF/human/genecentered_tss" \
            --num_epochs=100 \
            --warmup_frac=0.146 \
            --patience=25\
            --min_delta=0.00001 \
            --model_save_dir="gs://picard-testing-176520/enformer_batchnorms/models" \
            --model_save_basename="enformer_batchnorms" \
            --lr_base1="1.0e-04" \
            --lr_base2="1.0e-04" \
            --epsilon=1.0e-10 \
            --num_parallel=8 \
            --savefreq=8 \
            --val_examples_TSS=1646 \
            --load_init="False" \
            --freeze_conv_layers="False" \
            --enformer_checkpoint_path="sonnet_weights" \
            --num_examples_dict="human:34021,2213" \
            --model_type='enformer_performer'
            