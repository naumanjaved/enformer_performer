#!/bin/bash -l

python3 train_model.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="enformer_performer" \
            --wandb_user="njaved" \
            --wandb_sweep_name="enformer_performer" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://genformer_data/expanded_originals/196k" \
            --gcs_path_TSS="gs://genformer_data/expanded_originals/196k/human/tfrecords_tss" \
            --num_epochs=100 \
            --warmup_frac=0.146 \
            --patience=50\
            --min_delta=0.00001 \
            --model_save_dir="gs://picard-testing-176520/enformer_full/models" \
            --model_save_basename="enformer_performer_230209" \
            --lr_base="5.0e-04" \
            --gradient_clip="0.20" \
            --epsilon=1.0e-8 \
            --num_parallel=8 \
            --savefreq=5 \
            --val_examples_TSS=100 \
            --use_enformer_weights="False" \
            --num_examples_dict="human:34021,2213;mouse:29295,2209" \
            --heads_channels="human:5313;mouse:1643"
