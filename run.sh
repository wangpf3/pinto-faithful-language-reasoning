#!/bin/bash

project_dir='.'

dataset="csqa"
dev_split="dev"
test_split="train,dev,test.obqa,test.qasc"
sample_size=0
model_name='t5-base'
max_enc_length=128
max_dec_length=128
train_batch_size=16
eval_batch_size=32
grad_step=1
learning_rate=3e-4
weight_decay=0
num_epoch=10

dropout_context=0
label_smoothing_no_inference=0.1
mask_prob=0.5
counter_factor=1.0
mask_ratio=1.0
replace_ratio=0.3

save_dir="${project_dir}/checkpoints/${dataset}_${sample_size}-shot/dropout-context${dropout_context}_label-smooth${label_smoothing_no_inference}_mask${mask_ratio}-or-replace${replace_ratio}-inference${mask_prob}_${model_name}_bs${train_batch_size}_gs${grad_step}_lr${learning_rate}_wd${weight_decay}_e${num_epoch}"
mkdir -p $save_dir

python \
    main.py \
    --mask_ratio $mask_ratio \
    --replace_ratio $replace_ratio \
    --counter_factor $counter_factor \
    --mask_prob $mask_prob \
    --dropout_context $dropout_context \
    --label_smoothing_no_inference $label_smoothing_no_inference \
    --dataset $dataset \
    --sample_size $sample_size \
    --test_split $test_split \
    --dev_split $dev_split \
    --save_dir $save_dir \
    --model_name $model_name \
    --max_enc_length $max_enc_length \
    --max_dec_length $max_dec_length \
    --train_batch_size $train_batch_size \
    --eval_batch_size $eval_batch_size \
    --grad_step $grad_step \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --num_epoch $num_epoch \

