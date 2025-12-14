#!/bin/zsh

CUDA_VISIBLE_DEVICES=0,1 \
vllm serve /home/byguan/huggingface/Qwen/Qwen3-VL-32B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --allowed-local-media-path /home
    