#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=2

# DeepSpeed configuration
deepspeed=./utils/zero3.json

# Model configuration
llm=./huggingface/Qwen/Qwen2.5-VL-7B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=2e-7
batch_size=1
grad_accum_steps=4

# Training entry point
entry_file=./codes/train_qwen25vl_sft.py

# Dataset configuration (replace with public dataset names)
datasets="direct_finetune"

# Output configuration
run_name="qwen25vl-7b-sft"
timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
output_dir=./checkpoint/${timestamp}

# Create output directory if it doesn't exist
mkdir -p ${output_dir}

# Create log file to record all parameters
log_file="${output_dir}/training_params.log"
echo "Training Parameters Log - $(date)" > ${log_file}
echo "=================================" >> ${log_file}
echo "" >> ${log_file}
echo "Distributed Training Configuration:" >> ${log_file}
echo "MASTER_ADDR: ${MASTER_ADDR}" >> ${log_file}
echo "MASTER_PORT: ${MASTER_PORT}" >> ${log_file}
echo "NNODES: ${NNODES}" >> ${log_file}
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}" >> ${log_file}
echo "" >> ${log_file}
echo "Model Configuration:" >> ${log_file}
echo "DeepSpeed Config: ${deepspeed}" >> ${log_file}
echo "Model Path: ${llm}" >> ${log_file}
echo "" >> ${log_file}
echo "Training Hyperparameters:" >> ${log_file}
echo "Learning Rate: ${lr}" >> ${log_file}
echo "Batch Size: ${batch_size}" >> ${log_file}
echo "Gradient Accumulation Steps: ${grad_accum_steps}" >> ${log_file}
echo "" >> ${log_file}
echo "Dataset Configuration:" >> ${log_file}
echo "Datasets: ${datasets}" >> ${log_file}
echo "" >> ${log_file}
echo "Output Configuration:" >> ${log_file}
echo "Run Name: ${run_name}" >> ${log_file}
echo "Output Directory: ${output_dir}" >> ${log_file}
echo "Timestamp: ${timestamp}" >> ${log_file}
echo "" >> ${log_file}
echo "Entry File: ${entry_file}" >> ${log_file}
echo "" >> ${log_file}

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 2 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 50 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Log detailed training arguments
echo "Detailed Training Arguments:" >> ${log_file}
echo "DeepSpeed: ${deepspeed}" >> ${log_file}
echo "Model Name/Path: ${llm}" >> ${log_file}
echo "Dataset Use: ${datasets}" >> ${log_file}
echo "Data Flatten: True" >> ${log_file}
echo "Tune MM Vision: False" >> ${log_file}
echo "Tune MM MLP: True" >> ${log_file}
echo "Tune MM LLM: True" >> ${log_file}
echo "BF16: True" >> ${log_file}
echo "Output Dir: ${output_dir}" >> ${log_file}
echo "Num Train Epochs: 2" >> ${log_file}
echo "Per Device Train Batch Size: ${batch_size}" >> ${log_file}
echo "Per Device Eval Batch Size: $((batch_size*2))" >> ${log_file}
echo "Gradient Accumulation Steps: ${grad_accum_steps}" >> ${log_file}
echo "Max Pixels: 50176" >> ${log_file}
echo "Min Pixels: 784" >> ${log_file}
echo "Eval Strategy: no" >> ${log_file}
echo "Save Strategy: steps" >> ${log_file}
echo "Save Steps: 1000" >> ${log_file}
echo "Save Total Limit: 10" >> ${log_file}
echo "Learning Rate: ${lr}" >> ${log_file}
echo "Weight Decay: 0" >> ${log_file}
echo "Warmup Ratio: 0.03" >> ${log_file}
echo "Max Grad Norm: 1" >> ${log_file}
echo "LR Scheduler Type: cosine" >> ${log_file}
echo "Logging Steps: 1" >> ${log_file}
echo "Model Max Length: 16384" >> ${log_file}
echo "Gradient Checkpointing: True" >> ${log_file}
echo "Dataloader Num Workers: 4" >> ${log_file}
echo "Run Name: ${run_name}" >> ${log_file}
echo "Report To: wandb" >> ${log_file}
echo "" >> ${log_file}
echo "Complete Training Arguments (Raw):" >> ${log_file}
echo "${args}" >> ${log_file}
echo "" >> ${log_file}
echo "Training Command:" >> ${log_file}
echo "torchrun --nproc_per_node=${NPROC_PER_NODE} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} ${entry_file} ${args}" >> ${log_file}
echo "" >> ${log_file}
echo "Training started at: $(date)" >> ${log_file}
echo "=================================" >> ${log_file}

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        ${entry_file} ${args}