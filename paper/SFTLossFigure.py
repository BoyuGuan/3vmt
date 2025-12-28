import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def load_loss_data(file_path, loss_key='loss'):
    """
    读取json文件并提取step和指定的loss数据
    支持自定义 loss_key，并只提取 step <= 625 的数据
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return [], []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        log_history = data.get("log_history", [])
        steps = []
        losses = []
        
        for entry in log_history:
            if loss_key in entry and 'step' in entry:
                step_val = entry['step']
                
                # 截断超过 580 的步数
                if step_val > 580:
                    continue
                
                steps.append(step_val)
                losses.append(entry[loss_key])
                
        return steps, losses
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return [], []

def smooth_curve(data, window_size=40):
    """
    使用指数加权移动平均 (EMA) 平滑曲线
    相比简单的 rolling mean，EMA 得到的曲线更加平滑自然，且对近期趋势反应更好。
    window_size (span): 对应跨度，数值越大曲线越平滑。
    """
    if not data:
        return []
    series = pd.Series(data)
    # 使用 ewm (Exponential Weighted Moving Average) 替代 rolling
    # span 参数大致相当于移动平均的窗口大小，但权重呈指数衰减
    smoothed = series.ewm(span=window_size).mean()
    return smoothed.tolist()

def main():
    file_dart = r"./paper/src/qwen3-vl-4b-dart-sft-trainer-state.json"
    file_sft = r"./paper/src/qwen3-vl-4b-sft-trainer-state.json"
    
    output_dir = r"./paper/output"
    output_filename = "SFT_loss_comparison.pdf"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Loading data from (DART): {file_dart}")
    steps_dart, losses_dart = load_loss_data(file_dart, loss_key='translation_loss')
    
    print(f"Loading data from (SFT): {file_sft}")
    steps_sft, losses_sft = load_loss_data(file_sft, loss_key='loss')

    if not steps_dart or not steps_sft:
        print("Error: Could not load data from one or both files. Exiting.")
        return

    # --- 修改部分：增大平滑窗口 ---
    # 将窗口从 20 增加到 60，配合 ewm 方法可获得极佳的平滑度
    smooth_window = 30
    smooth_losses_dart = smooth_curve(losses_dart, window_size=smooth_window)
    smooth_losses_sft = smooth_curve(losses_sft, window_size=smooth_window)

    plt.style.use('seaborn-v0_8-whitegrid') 
    
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)


    # 绘制 SFT
    color_sft = "#FF7F0E"
    ax.plot(steps_sft, losses_sft, color=color_sft, alpha=0.12, linewidth=1) 
    ax.plot(steps_sft, smooth_losses_sft, color=color_sft, alpha=0.95, linewidth=2.5, 
            label='SFT (Baseline)')
    # 绘制 DART
    color_dart = "#1F77B4"
    ax.plot(steps_dart, losses_dart, color=color_dart, alpha=0.12, linewidth=1) 
    ax.plot(steps_dart, smooth_losses_dart, color=color_dart, alpha=0.95, linewidth=2.5, 
            label='DART SFT (Ours)')

    ax.set_title('Translation Tokens Loss Comparison', fontsize=18, fontweight='bold', pad=20)
    # --- 字体与样式设置 ---
    ax.set_xlabel('Global Step', fontsize=16, labelpad=10, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=16, labelpad=10, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # max_val = max(max(smooth_losses_dart), max(smooth_losses_sft)) * 1.2
    ax.set_ylim(0, 3.5)
    ax.set_xlim(0, 600) 
    
    ax.legend(
        fontsize=14, 
        frameon=True, 
        fancybox=True, 
        framealpha=0.9, 
        loc='upper right',
        prop={'weight': 'bold', 'size': 14} 
    )
    
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='#e0e0e0')
    ax.spines['left'].set_linewidth(1.5) 
    ax.spines['bottom'].set_linewidth(1.5) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    
    print(f"Success! Figure saved to: {output_path}")

if __name__ == "__main__":
    main()