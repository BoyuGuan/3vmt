"""
将多模态信息与翻译句子融合，生成不同类型的 prompt。

功能：
1. 读取 extractMM.py 的输出文件（包含 MMInfo 字段）
2. 根据每条数据的 language 字段自动判断翻译方向（en->zh 或 zh->en）
3. 为每种 cue 类型生成对应的 prompt
4. 将所有 prompt 存储到原始数据中，输出为单个 JSON 文件
"""

from __future__ import annotations

from typing import Any, Dict, List


# -----------------------------
# System prompt (English only)
# -----------------------------

SYSTEM_PROMPT = (
    "You are a professional translator.\n"
    "Translate the source sentence into the target language.\n"
    "Use the provided video cue ONLY as auxiliary context when it helps disambiguate meaning.\n"
    "Do NOT invent any details beyond the cue.\n"
    "Output ONLY the final translation, with no explanation."
)

# All supported cue types
ALL_CUE_TYPES = ["baseline", "people", "objects", "actions", "ocr", "spatial_relations", "pointing_gaze", "all_cues"]


def parse_language_field(language_str: str) -> tuple:
    """
    解析 language 字段，返回 (src_lang_code, src_lang_name, tgt_lang_code, tgt_lang_name)
    
    language 字段格式: "en: English" 或 "zh: Chinese"
    """
    lang_code = language_str.split(":")[0].strip().lower() if language_str else "en"
    
    if lang_code == "en":
        return ("en", "English", "zh", "Chinese")
    elif lang_code == "zh":
        return ("zh", "Chinese", "en", "English")
    else:
        # 默认 en -> zh
        return ("en", "English", "zh", "Chinese")


def build_all_cue_type_prompts(
    extraction: Dict[str, Any],
    source_sentence: str,
    src_lang_name: str = "English",
    tgt_lang_name: str = "Chinese",
) -> Dict[str, str]:
    """
    为每种 cue 类型生成一个合并后的 prompt 文本。
    
    返回: Dict[cue_type, prompt_text]
    """
    people = extraction.get("people", []) or []
    objects = extraction.get("objects", []) or []
    actions = extraction.get("actions", []) or []
    ocr = extraction.get("ocr", []) or []
    spatial = extraction.get("spatial_relations", []) or []
    gaze = extraction.get("pointing_gaze", []) or []
    
    def _fmt_kv(attrs: Dict[str, Any]) -> str:
        if not isinstance(attrs, dict) or not attrs:
            return ""
        chunks = []
        for k, v in attrs.items():
            if v is None or (isinstance(v, str) and not v.strip()):
                continue
            chunks.append(f"{k}={v}")
        return ", ".join(chunks)
    
    # Verbalizers for each cue type
    def verb_people(items: List[Dict]) -> str:
        lines = []
        for it in items:
            if not isinstance(it, dict):
                continue
            pid = it.get("person_id", "unknown")
            role = (it.get("role_guess") or {}).get("role", "unknown")
            lines.append(f"- {pid}: role guess = {role}")
        return "\n".join(lines) if lines else ""
    
    def verb_objects(items: List[Dict]) -> str:
        lines = []
        for it in items:
            if not isinstance(it, dict):
                continue
            oid = it.get("object_id", "unknown")
            cat = it.get("category", "unknown")
            attrs = _fmt_kv(it.get("attributes", {}))
            line = f"- {oid}: {cat}"
            if attrs:
                line += f" [{attrs}]"
            lines.append(line)
        return "\n".join(lines) if lines else ""
    
    def verb_actions(items: List[Dict]) -> str:
        lines = []
        for it in items:
            if not isinstance(it, dict):
                continue
            aid = it.get("action_id", "unknown")
            pred = it.get("predicate", "unknown")
            agent = it.get("agent_id", "unknown")
            patient = it.get("patient_id", "unknown")
            inst = it.get("instrument_id", "unknown")
            lines.append(f"- {aid}: {pred} (agent={agent}, patient={patient}, instrument={inst})")
        return "\n".join(lines) if lines else ""
    
    def verb_ocr(items: List[Dict]) -> str:
        lines = []
        for it in items:
            if not isinstance(it, dict):
                continue
            text = it.get("text", "")
            if text:
                lines.append(f"- \"{text}\"")
        return "\n".join(lines) if lines else ""
    
    def verb_spatial(items: List[Dict]) -> str:
        lines = []
        for it in items:
            if not isinstance(it, dict):
                continue
            subj = it.get("subject_id", "unknown")
            rel = it.get("relation", "unknown")
            obj = it.get("object_id", "unknown")
            lines.append(f"- {subj} is {rel} {obj}")
        return "\n".join(lines) if lines else ""
    
    def verb_gaze(items: List[Dict]) -> str:
        lines = []
        for it in items:
            if not isinstance(it, dict):
                continue
            src = it.get("source_id", "unknown")
            typ = it.get("type", "unknown")
            tgt = it.get("target_id", "unknown")
            desc = it.get("target_description", "")
            tgt_str = f"{tgt} ({desc})" if desc and tgt == "unknown" else tgt
            lines.append(f"- {src} {typ} -> {tgt_str}")
        return "\n".join(lines) if lines else ""
    
    # Build cue texts for each type
    cue_texts = {}
    
    # Baseline (no cue)
    cue_texts["baseline"] = "No video cue is provided. Translate using text only."
    
    # Individual cue types
    cue_texts["people"] = f"Who-is-who (people in video):\n{verb_people(people)}" if people else ""
    cue_texts["objects"] = f"Objects visible in video:\n{verb_objects(objects)}" if objects else ""
    cue_texts["actions"] = f"Actions observed in video:\n{verb_actions(actions)}" if actions else ""
    cue_texts["ocr"] = f"On-screen text (OCR):\n{verb_ocr(ocr)}" if ocr else ""
    cue_texts["spatial_relations"] = f"Spatial relations:\n{verb_spatial(spatial)}" if spatial else ""
    cue_texts["pointing_gaze"] = f"Pointing/Gaze directions:\n{verb_gaze(gaze)}" if gaze else ""
    
    # All cues combined
    all_cue_parts = [v for k, v in cue_texts.items() if k != "baseline" and v]
    cue_texts["all_cues"] = "Video cues:\n" + "\n\n".join(all_cue_parts) if all_cue_parts else "No video cues available."
    
    # Compose full prompts
    prompts = {}
    for cue_type, cue_text in cue_texts.items():
        if not cue_text:
            # Empty cue, use baseline
            cue_text = cue_texts["baseline"]
        
        user_prompt = (
            f"Task: Translate from {src_lang_name} to {tgt_lang_name}.\n\n"
            f"Source sentence:\n{source_sentence}\n\n"
            f"Video cue (use only if helpful):\n{cue_text}\n"
        )
        
        prompts[cue_type] = user_prompt
    
    return prompts


# -----------------------------
# Main program for batch processing
# -----------------------------
if __name__ == "__main__":
    import argparse
    import json
    import os
    import logging
    from collections import defaultdict
    
    logger = logging.getLogger('makeMMPrompt')
    formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser(description="将多模态信息与翻译句子融合生成不同类型的 prompt")
    parser.add_argument("--inputFilePath", type=str, required=True,
                        help="输入文件路径（extractMM.py 的输出，包含 MMInfo 字段）")
    parser.add_argument("--outputFilePath", type=str, default="./data/work3/MMPrompts/data_with_prompts.json",
                        help="输出文件路径")
    parser.add_argument("--cueTypes", type=str, default="all",
                        help="要生成的 cue 类型，逗号分隔（baseline,people,objects,actions,ocr,spatial_relations,pointing_gaze,all_cues），或 'all' 表示全部")
    parser.add_argument("--logDir", type=str, default="./log",
                        help="日志目录")
    parser.add_argument("--onlyValidMMInfo", action="store_true",
                        help="是否只处理 MMInfoValid=True 的数据")
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs(args.logDir, exist_ok=True)
    fileHandler = logging.FileHandler(f'{args.logDir}/makeMMPrompt.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    # Determine cue types to generate
    if args.cueTypes.lower() == "all":
        selected_cue_types = ALL_CUE_TYPES
    else:
        selected_cue_types = [t.strip() for t in args.cueTypes.split(",")]
        for t in selected_cue_types:
            if t not in ALL_CUE_TYPES:
                logger.error(f"未知的 cue 类型: {t}")
                raise ValueError(f"未知的 cue 类型: {t}. 可选: {ALL_CUE_TYPES}")
    
    logger.info(f"输入文件: {args.inputFilePath}")
    logger.info(f"输出文件: {args.outputFilePath}")
    logger.info(f"要生成的 cue 类型: {selected_cue_types}")
    
    # Read input file
    try:
        with open(args.inputFilePath, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        logger.info(f"成功读取 {len(input_data)} 条数据")
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        raise
    
    # Statistics
    total_count = 0
    processed_count = 0
    skipped_no_mminfo = 0
    lang_direction_counts = defaultdict(int)  # 统计翻译方向
    cue_type_counts = defaultdict(int)  # 每种 cue 类型有多少条数据有非空 cue
    
    # Output data
    output_data = []
    
    # Process each item
    for idx, item in enumerate(input_data):
        total_count += 1
        
        # Check if we should skip invalid MMInfo
        mm_info_valid = item.get("MMInfoValid", False)
        if args.onlyValidMMInfo and not mm_info_valid:
            skipped_no_mminfo += 1
            continue
        
        # Parse language field to determine translation direction
        language_field = item.get("language", "en: English")
        src_lang_code, src_lang_name, tgt_lang_code, tgt_lang_name = parse_language_field(language_field)
        
        # Track language direction statistics
        lang_direction = f"{src_lang_code}->{tgt_lang_code}"
        lang_direction_counts[lang_direction] += 1
        
        # Get source sentence based on detected source language
        src_sent_field = f"{src_lang_code.upper()}_sentence"
        src_sentence = item.get(src_sent_field, "")
        if not src_sentence:
            logger.warning(f"第 {idx+1} 条数据没有 {src_sent_field} 字段")
            continue
        
        # Get MMInfo
        mm_info = item.get("MMInfo", {})
        if not isinstance(mm_info, dict):
            mm_info = {}
        
        # Generate prompts for each cue type
        prompts = build_all_cue_type_prompts(
            extraction=mm_info,
            source_sentence=src_sentence,
            src_lang_name=src_lang_name,
            tgt_lang_name=tgt_lang_name,
        )
        
        # Create output item: copy original data and add prompts
        output_item = item.copy()
        
        # Add translation direction info
        output_item["src_lang"] = src_lang_code
        output_item["tgt_lang"] = tgt_lang_code
        
        # Add prompts for each selected cue type as separate fields
        # Format: mm_prompt_<cue_type>
        for cue_type in selected_cue_types:
            prompt_text = prompts.get(cue_type, prompts["baseline"])
            output_item[f"mm_prompt_{cue_type}"] = prompt_text
            
            # Track cue type statistics
            has_content = False
            if cue_type == "baseline":
                has_content = True
            elif cue_type == "all_cues":
                has_content = any(mm_info.get(k) for k in ["people", "objects", "actions", "ocr", "spatial_relations", "pointing_gaze"])
            else:
                has_content = bool(mm_info.get(cue_type))
            
            if has_content:
                cue_type_counts[cue_type] += 1
        
        output_data.append(output_item)
        processed_count += 1
        
        # Progress
        if (idx + 1) % 1000 == 0:
            logger.info(f"已处理 {idx+1}/{len(input_data)} 条数据")
    
    # Create output directory
    output_dir = os.path.dirname(args.outputFilePath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save output
    try:
        with open(args.outputFilePath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存 {len(output_data)} 条数据到: {args.outputFilePath}")
    except Exception as e:
        logger.error(f"保存文件失败: {e}")
        raise
    
    # Print statistics
    separator = "=" * 70
    stats_output = f"""
{separator}
处理完成！统计信息：
{separator}

【整体统计】
  总数据量: {total_count}
  已处理数据量: {processed_count}
  跳过（MMInfo无效）: {skipped_no_mminfo}

【翻译方向统计】
"""
    
    for direction, count in sorted(lang_direction_counts.items()):
        pct = count / processed_count * 100 if processed_count > 0 else 0
        stats_output += f"  - {direction}: {count} ({pct:.2f}%)\n"
    
    stats_output += f"""
【各 cue 类型有效数据量】（有非空 cue 内容的数据）
"""
    
    for cue_type in selected_cue_types:
        count = cue_type_counts[cue_type]
        pct = count / processed_count * 100 if processed_count > 0 else 0
        stats_output += f"  - {cue_type:20s}: {count:6d} ({pct:.2f}%)\n"
    
    stats_output += f"""
【输出文件】
  路径: {args.outputFilePath}
  数据条数: {len(output_data)}
  每条数据包含的 prompt 字段:
"""
    for cue_type in selected_cue_types:
        stats_output += f"    - mm_prompt_{cue_type}\n"
    
    stats_output += f"\n{separator}\n"
    
    logger.info(stats_output)
    print(stats_output)
    
    # Save statistics
    stats_file = args.outputFilePath.replace('.json', '_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(stats_output)
    logger.info(f"统计信息已保存到: {stats_file}")
