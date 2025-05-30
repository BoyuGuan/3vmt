#!/bin/zsh

export CUDA_VISIBLE_DEVICES=1

# 定义基础目录数组
base_dirs=("/home/byguan/LLMvmt/eval/eval-2025-05-28" "/home/byguan/LLMvmt/eval/eval-2025-05-29")

# 循环遍历基础目录
for base_dir_prefix in "${base_dirs[@]}"; do
  # 查找所有以指定前缀开头的文件夹
  find /home/byguan/LLMvmt/eval/ -maxdepth 1 -type d -name "$(basename "$base_dir_prefix")*" -print0 | while IFS= read -r -d $'\0' dir; do
    # 检查文件夹中是否包含 results.json 文件
    if [ -f "${dir}/results.json" ]; then
      echo "找到 results.json 于: ${dir}"
      # 对该文件夹运行指定的Python脚本
      echo "运行命令: CUDA_VISIBLE_DEVICES=1 python3 ./utils/computeTransMetric.py --dir_path ${dir}"
      CUDA_VISIBLE_DEVICES=1 python3 ./utils/computeTransMetric.py --dir_path "${dir}"
      echo "命令执行完毕: ${dir}"
      echo "--------------------------------------------------"
    fi
  done
done

echo "所有符合条件的文件夹处理完毕。"