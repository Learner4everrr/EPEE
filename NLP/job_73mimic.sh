#!/bin/bash
#SBATCH --nodes=1
#SBATCH -p dos-chs,a100-4,agsmall,ag2tb,a100-8,amdsmall,amdlarge,amd512,amd2tb
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50g
#SBATCH --time=4:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhan8023@umn.edu

# module load cuda/11.2
source ~/.bashrc
conda activate /home/zhan1386/zhan8023/test/earlyexit/PCEE-BERT/.env/

nvidia-smi
# sh scripts/mrpc/train.sh > result

task_name=cola
metric_key=acc
data_dir=MIMIC
max_seq_length=512
model_type=albert
model_name=albert

epoch_num=15
seed=112411
l_rate=2e-4
echo "${task_name}"
echo "${seed}"
echo "${l_rate}"

# Train
CUDA_VISIBLE_DEVICES="0" python changefile.py --dataset_name ${data_dir}

CUDA_VISIBLE_DEVICES="0" python -u run_glue_with_pabee.py \
    --seed ${seed} \
    --task_name ${task_name} \
    --data_dir datasets/glue/${data_dir}/ \
    --model_type ${model_type} \
    --model_name_or_path ${model_name} \
    --cache_dir ./distilbert \
    --output_dir experiments/outputs/${model_type}_${task_name}_${data_dir} \
    --max_seq_length ${max_seq_length} \
    --do_train --do_eval\
    --evaluate_during_training \
    --do_lower_case \
    --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 64 \
    --learning_rate ${l_rate} \
    --linear_learning_rate ${l_rate} \
    --num_train_epochs ${epoch_num} \
    --warmup_steps 200 --logging_steps 100 --save_steps 100 \
    --eval_all_checkpoints \
    --overwrite_output_dir --overwrite_cache \
    --gradient_checkpointing \
    --weights_schema asc \
    --patience -1 \
    --ee_mechanism Ve \
    --metric_key ${metric_key}


# entropy
CUDA_VISIBLE_DEVICES="0" python changefile.py --dataset_name ${data_dir}
patiences=1
exiting_threshold=(0.001 0.005 0.01 0.03 0.05 0.1)
for e_thres in ${exiting_threshold[@]}
do
    CUDA_VISIBLE_DEVICES="0" python -u run_glue_with_pabee.py \
        --seed ${seed} \
        --task_name $task_name \
        --data_dir datasets/glue/${data_dir}/ \
        --model_type ${model_type} \
        --model_name_or_path ${model_name} \
        --cache_dir ./distilbert \
        --output_dir experiments/outputs/${model_type}_${task_name}_${data_dir} \
        --max_seq_length ${max_seq_length} \
        --do_eval \
        --evaluate_during_training \
        --do_lower_case \
        --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 1 \
        --learning_rate ${l_rate} \
        --linear_learning_rate ${l_rate} \
        --num_train_epochs ${epoch_num} \
        --warmup_steps 200 --logging_steps 100 --save_steps 100 \
        --eval_all_checkpoints \
        --overwrite_output_dir --overwrite_cache \
        --gradient_checkpointing \
        --weights_schema asc \
        --patience ${patiences} \
        --exiting_threshold ${e_thres} \
        --ee_mechanism Ve \
        --metric_key ${metric_key}
done

# patience
CUDA_VISIBLE_DEVICES="0" python changefile.py --dataset_name ${data_dir}
patiences=1,2,3,4,5,6,7,8,9,10,11
e_thres=0
CUDA_VISIBLE_DEVICES="0" python -u run_glue_with_pabee.py \
    --seed ${seed} \
    --task_name $task_name \
    --data_dir datasets/glue/${data_dir}/ \
    --model_type ${model_type} \
    --model_name_or_path ${model_name} \
    --cache_dir ./distilbert \
    --output_dir experiments/outputs/${model_type}_${task_name}_${data_dir} \
    --max_seq_length ${max_seq_length} \
    --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 1 \
    --learning_rate ${l_rate} \
    --linear_learning_rate ${l_rate} \
    --num_train_epochs ${epoch_num} \
    --warmup_steps 200 --logging_steps 100 --save_steps 100 \
    --eval_all_checkpoints \
    --overwrite_output_dir --overwrite_cache \
    --gradient_checkpointing \
    --weights_schema asc \
    --patience ${patiences} \
    --exiting_threshold ${e_thres} \
    --ee_mechanism V0 \
    --metric_key ${metric_key}


# EPEE
CUDA_VISIBLE_DEVICES="0" python changefile.py --dataset_name ${data_dir}
patiences=1,2,3,4,5,6,7,8,9,10,11
exiting_threshold=(0.001 0.005 0.01 0.03 0.05 0.1)
for e_thres in ${exiting_threshold[@]}
do
    CUDA_VISIBLE_DEVICES="0" python -u run_glue_with_pabee.py \
        --seed ${seed} \
        --task_name $task_name \
        --data_dir datasets/glue/${data_dir}/ \
        --model_type ${model_type} \
        --model_name_or_path ${model_name} \
        --cache_dir ./distilbert \
        --output_dir experiments/outputs/${model_type}_${task_name}_${data_dir} \
        --max_seq_length ${max_seq_length} \
        --do_eval \
        --evaluate_during_training \
        --do_lower_case \
        --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 1 \
        --learning_rate ${l_rate} \
        --linear_learning_rate ${l_rate} \
        --num_train_epochs ${epoch_num} \
        --warmup_steps 200 --logging_steps 100 --save_steps 100 \
        --eval_all_checkpoints \
        --overwrite_output_dir --overwrite_cache \
        --gradient_checkpointing \
        --weights_schema asc \
        --patience ${patiences} \
        --exiting_threshold ${e_thres} \
        --ee_mechanism V0e \
        --metric_key ${metric_key}
done