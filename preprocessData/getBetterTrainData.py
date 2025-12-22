# -*- coding: utf-8 -*-
# 同时保存SFT数据和对应的源数据

import argparse
import json
import os
import random
import logging
from typing import Dict, Any

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./log/process_metrics.log", mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 需要排除的字段
EXCLUDED_FIELDS = {"translation_baseline", "translation_all_cues"}

def get_clean_cue_content(item: Dict[str, Any], cue_type: str) -> str:
    """
    从 mm_prompt_{cue_type} 中提取纯净的视频信息内容。
    """
    prompt_key = f"mm_prompt_{cue_type.replace('translation_', '')}"
    full_prompt = item.get(prompt_key, "")
    
    if not full_prompt:
        return "No information provided."

    if "Video cue:" in full_prompt:
        content = full_prompt.split("Video cue:")[-1].strip()
        return content
    
    return full_prompt

def generate_sft_item(item: Dict[str, Any], 
                      sft_type: str, 
                      cue_key: str = None, 
                      cue_content: str = None) -> Dict[str, Any]:
    """
    构造符合要求的 SFT 数据格式。
    修改：使用 Reference (Ref) 作为 Target 输出。
    """
    video_id = item['video_id']
    clip_id = item['clip_id']

    # 确定源语言和目标语言
    # 优先使用 explicit 的字段，如果没有则根据 language 字段推断
    src_lang = item.get('src_lang', '').strip().lower()
    
    if not src_lang:
        lang_field = item.get('language', '').lower()
        if lang_field.startswith('en'):
            src_lang = 'en'
        elif lang_field.startswith('zh'):
            src_lang = 'zh'
        else:
            raise ValueError(f"Cannot determine source language for item with video_id: {video_id}")

    # 根据源语言决定 Source 和 Target(Ref)
    if src_lang == 'zh':
        src_sentence = item.get('ZH_sentence', '')
        # 如果源是中文，Ref(Target) 应该是英文
        target_ref = item.get('EN_sentence', '')
    else:
        src_sentence = item.get('EN_sentence', '')
        # 如果源是英文，Ref(Target) 应该是中文
        target_ref = item.get('ZH_sentence', '')

    # --- 核心修改：GPT 的回答现在总是 Reference ---
    target_output = target_ref

    if sft_type == "baseline":
        # 题目要求：翻译这个样本【不需要】视频信息...
        human_prompt = (
            f"<video>\n"
            f"Translate this sample **without** video information.\n"
            f"Input sentence:\n{src_sentence}\n"
            f"Translated sentence:\n"
        )

    else:
        # Visual Positive Sample
        cue_name = cue_key.replace("translation_", "")
        # 题目要求：翻译这个样本【需要】视频信息...
        human_prompt = (
            f"<video>\n"
            f"Translate this sample **with** video information [{cue_name}].\n"
            f"It is:\n{cue_content}\n"
            # f"So the combined translation is:\n"
            f"Input sentence:\n{src_sentence}\n"
            f"Translated sentence:\n"
        )

    # 构造 Video 路径
    video_path = f"./data/TriFine/videoClips/{video_id}/{video_id}_{clip_id}.mp4"

    return {
        "video": video_path,
        "conversations": [
            {
                "from": "human",
                "value": human_prompt
            },
            {
                "from": "gpt",
                "value": target_output  # 这里现在是 Ref
            }
        ]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="./data/work3/MMinfoAndTrans/promptsAndTransMetrics.json")
    parser.add_argument("--output_json", type=str, default="./data/work3/sftData/sft_MMInfo_train_data.json")
    parser.add_argument("--output_source_json", type=str, default="./data/work3/meta_train_data.json", help="Save corresponding source items from input_json")
    parser.add_argument("--alpha", type=float, default=0.6, help="Ratio of visual-enhanced samples (alpha) to baseline samples (1-alpha)")
    parser.add_argument("--comet_diff", type=float, default=2.0, help="Threshold for COMET improvement over baseline")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(args.input_json):
        raise FileNotFoundError(f"Input file not found: {args.input_json}")

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} items from {args.input_json}")

    # --- 任务 1: 统计 diff > comet_diff 的样本数（同一样本只计一次） ---
    count_diff_gt_1 = 0
    
    # --- 任务 2 准备: 筛选候选集 ---
    # Group A: Visual Candidates (COMET > Baseline + comet_diff)
    group_a_candidates = [] 

    # Group B: Baseline Candidates (Baseline is Max among all excluding all_cues)
    group_b_candidates = []

    for item in data:
        metrics = item.get("translation_metrics", {})
        if not metrics:
            continue

        baseline_metrics = metrics.get("translation_baseline")
        if not baseline_metrics:
            continue
        
        baseline_comet = baseline_metrics["COMET"]
        
        # 获取所有除 baseline 和 all_cues 之外的有效 visual keys
        visual_keys = [k for k in metrics.keys() if k not in EXCLUDED_FIELDS]
        
        # 1. 筛选 Group A: COMET > Baseline + comet_diff
        better_cues = []
        for k in visual_keys:
            score = metrics[k]["COMET"]
            if score > baseline_comet + args.comet_diff:
                better_cues.append((k, score))

        # 任务 1 统计：同一个样本只计一次
        if better_cues:
            count_diff_gt_1 += 1
        
        if better_cues:
            max_score = max(c[1] for c in better_cues)
            best_cues_for_item = [c for c in better_cues if c[1] == max_score]
            
            for cue_key, score in best_cues_for_item:
                group_a_candidates.append({
                    "item": item,
                    "cue_key": cue_key,
                    "score": score
                })
        
        # 2. 筛选 Group B: Baseline is Highest
        is_baseline_highest = True
        for k in visual_keys:
            if metrics[k]["COMET"] > baseline_comet:
                is_baseline_highest = False
                break
        
        if is_baseline_highest:
            group_b_candidates.append({
                "item": item,
                "score": baseline_comet
            })

    # 输出统计结果
    logger.info(f"-" * 30)
    logger.info(
        f"Task 1 Result: Total samples where a visual cue COMET > Baseline + {args.comet_diff}: {count_diff_gt_1}"
    )
    logger.info(f"-" * 30)

    # --- 任务 2 采样逻辑 ---
    len_a = len(group_a_candidates)
    len_b = len(group_b_candidates)
    
    logger.info(f"Candidates Found -> Group A (Visual > Base+{args.comet_diff}): {len_a}")
    logger.info(f"Candidates Found -> Group B (Baseline Highest): {len_b}")

    if args.alpha <= 0 or args.alpha >= 1:
        logger.error("Alpha must be between 0 and 1.")
        return

    # 计算采样数
    max_total_by_a = len_a / args.alpha
    max_total_by_b = len_b / (1.0 - args.alpha)
    target_total = min(max_total_by_a, max_total_by_b)
    
    target_count_a = int(target_total * args.alpha)
    target_count_b = int(target_total * (1.0 - args.alpha))

    sampled_a = random.sample(group_a_candidates, target_count_a) if len_a >= target_count_a else group_a_candidates
    sampled_b = random.sample(group_b_candidates, target_count_b) if len_b >= target_count_b else group_b_candidates

    logger.info(f"Sampling Target -> Total: {int(target_total)} | Group A: {len(sampled_a)} | Group B: {len(sampled_b)}")

    # --- 任务 3: 构造 SFT 数据 ---
    sft_pairs = []

    # 处理 Group A (Visual Enhanced)
    for obj in sampled_a:
        item = obj['item']
        cue_key = obj['cue_key']
        cue_content = get_clean_cue_content(item, cue_key)
        
        sft_item = generate_sft_item(
            item=item,
            sft_type="visual",
            cue_key=cue_key,
            cue_content=cue_content
        )
        sft_pairs.append((sft_item, item))

    # 处理 Group B (Baseline Best)
    for obj in sampled_b:
        item = obj['item']
        sft_item = generate_sft_item(
            item=item,
            sft_type="baseline"
        )
        sft_pairs.append((sft_item, item))

    random.shuffle(sft_pairs)

    sft_data = [pair[0] for pair in sft_pairs]
    source_data = [pair[1] for pair in sft_pairs]

    output_source_json = args.output_source_json

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=4)

    logger.info(f"SFT data (using REF as target) saved to: {args.output_json}")

    source_dir = os.path.dirname(output_source_json)
    if source_dir:
        os.makedirs(source_dir, exist_ok=True)
    with open(output_source_json, "w", encoding="utf-8") as f:
        json.dump(source_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Source items saved to: {output_source_json}")

if __name__ == "__main__":
    main()
