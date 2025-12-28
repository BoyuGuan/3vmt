#!/bin/bash
# GRPO Training Script for Qwen3-VL-4B (Optimized for A100-80GB x4 & Video Data)

set -x

# 默认使用 sglang 引擎，因为 vLLM V1 对 Qwen3-VL 视频处理有 bug
ENGINE=${1:-sglang}

# === 数据路径 ===
TRAIN_DATA=${TRAIN_DATA:-"$HOME/3vmt/data/work3/rl_data/train.parquet"}
VAL_DATA=${VAL_DATA:-"$HOME/3vmt/data/work3/rl_data/val.parquet"}

# === 模型路径 ===
MODEL_PATH=${MODEL_PATH:-"$HOME/3vmt/qwen3-vl-finetune/output/DART-SFT-Qwen3VL-4B"}

# === 输出路径 ===
PROJECT_NAME=${PROJECT_NAME:-"verl_grpo_vmt"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3_vl_4b_vmt_dart_GRPO"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"$HOME/3vmt/checkpoint/$(date +%Y-%m-%d-%H-%M-%S)"}

# === 关键训练参数 (针对视频显存优化) ===
# 1. 总Batch Size: 增大以提高GPU利用率
#    公式: Global_Batch = Micro_Batch * GPU数量 * 梯度累积步数
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}  # 从64增加到128

# 2. 序列长度
#    10s视频约占 3000-5000 tokens，加上翻译文本，8192通常够用
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

# 3. 学习率与Epoch
LEARNING_RATE=${LEARNING_RATE:-5e-7} # 4B模型微调建议稍微调小LR
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}      # 先跑5个epoch看效果

# 4. GRPO 采样数
#    每条数据生成多少个答案。对于翻译任务，4-8通常合适
N_ROLLOUTS=${N_ROLLOUTS:-4}

# === GPU 硬件配置 ===
N_GPUS=${N_GPUS:-4}
TP_SIZE=${TP_SIZE:-1} # 4B 模型单卡能放下，TP=1 效率最高

# === 性能优化参数 ===
# Micro batch sizes - 增大以提高吞吐量
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}     
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-2}    
ROLLOUT_MICRO_BATCH=${ROLLOUT_MICRO_BATCH:-4}      
REF_MICRO_BATCH=${REF_MICRO_BATCH:-4}              

# GPU显存利用率 - A100-80GB可以设得更高
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.8}           

mkdir -p "$CHECKPOINT_DIR"

# === 环境优化 ===
# 启用 CUDA 优化
export CUDA_DEVICE_MAX_CONNECTIONS=1
# 启用 TF32 以加速矩阵运算（A100支持）
export NVIDIA_TF32_OVERRIDE=1
# 优化 NCCL 通信
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
# PyTorch 优化
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === 启动命令 ===
# 优化后的配置：提高显存利用率和训练速度

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=False \
    data.truncation='right' \
    data.video_key=video \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_MICRO_BATCH \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=$ENGINE \
    \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
    \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.n=$N_ROLLOUTS \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_MICRO_BATCH \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    $@