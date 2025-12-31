#!/bin/bash

# 定义基准时间戳
THRESHOLD="eval-2025-12-27-00-44-59"

# eval 目录路径
EVAL_DIR="./eval"

# 检查 eval 目录是否存在
if [ ! -d "$EVAL_DIR" ]; then
    echo "错误: $EVAL_DIR 目录不存在"
    exit 1
fi

# 遍历 eval 目录下的所有子目录
for dir in "$EVAL_DIR"/eval-*; do
    # 检查是否是目录
    if [ -d "$dir" ]; then
        # 提取目录名
        dirname=$(basename "$dir")
        
        # 比较时间戳（字符串比较）
        if [[ "$dirname" > "$THRESHOLD" || "$dirname" == "$THRESHOLD" ]]; then
            echo "处理目录: $dir"
            python3 ./utils/computeTransMetric.py -d "$dir"
            
            # 检查命令执行状态
            if [ $? -eq 0 ]; then
                echo "✓ 完成: $dir"
            else
                echo "✗ 失败: $dir"
            fi
            echo "---"
        fi
    fi
done

echo "所有处理完成！"
