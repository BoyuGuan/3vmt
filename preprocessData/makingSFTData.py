import argparse
import logging
import json
import random
import os

import numpy as np

from utils.prompts import getUserPrompt

logger = logging.getLogger('cleanTrainData')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

def loadVideoCaption(srcLanguage):
    """加载video caption数据"""
    videoCaption = []
    if srcLanguage == "en":
        caption_files = [
            "./data/work3/videoCaption/Qwen2.5-VL-7B/eval-2025-06-07-14-38-06/results.json",
            "./data/work3/videoCaption/Qwen2.5-VL-7B/eval-2025-06-08-01-57-05/results.json",
            "./data/work3/videoCaption/Qwen2.5-VL-7B/eval-2025-06-08-13-16-27/results.json",
            "./data/work3/videoCaption/Qwen2.5-VL-7B/eval-2025-06-13-16-22-46/results.json",
            "./data/work3/videoCaption/Qwen2.5-VL-7B/eval-2025-06-13-16-22-43/results.json",
            "./data/work3/videoCaption/Qwen2.5-VL-7B/eval-2025-06-09-21-00-58/results.json",
            "./data/work3/videoCaption/Qwen2.5-VL-7B/eval-2025-06-18-16-14-20/results.json",
            "./data/work3/videoCaption/Qwen2.5-VL-7B/eval-2025-06-19-10-28-58/results.json"
        ]
    elif srcLanguage == "zh":
        caption_files = [
            "./data/work3/videoCaption/Qwen2.5-VL-7B/eval-2025-06-07-14-39-14/results.json",
            "./data/work3/videoCaption/Qwen2.5-VL-7B/eval-2025-06-08-01-51-58/results.json",
            "./data/work3/videoCaption/Qwen2.5-VL-7B/eval-2025-06-08-13-06-37/results.json",
            "./data/work3/videoCaption/Qwen2.5-VL-7B/eval-2025-06-09-00-21-06/results.json",
            "./data/work3/videoCaption/Qwen2.5-VL-7B/eval-2025-06-09-21-00-58/results.json",
            "./data/work3/videoCaption/Qwen2.5-VL-7B/eval-2025-06-18-16-14-29/results.json",
            "./data/work3/videoCaption/Qwen2.5-VL-7B/eval-2025-06-19-10-47-11/results.json"
        ]
    else:
        raise ValueError(f"srcLanguage {srcLanguage} not supported")
    
    for file_path in caption_files:
        if os.path.exists(file_path):
            videoCaption += json.load(open(file_path, "r"))
    
    return {item["clipID"]: item["preds"] for item in videoCaption}

def filterAndMatchClips(clips, scores, alpha, number, srcLanguage, tgtLanguage):
    """过滤clips并匹配video caption，确保最终有number条有caption的数据"""
    
    # 加载video caption
    clipID2VideoCaption = loadVideoCaption(srcLanguage)
    
    # 按分数排序，从高到低
    clips_with_scores = list(zip(clips, scores))
    clips_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 过滤出分数高于阈值的clips
    high_score_clips = [clip for clip, score in clips_with_scores if score > alpha]
    
    # 分离有caption和没有caption的clips
    clips_with_caption = []
    clips_without_caption = []
    
    for clip in high_score_clips:
        clip_id = f'{clip["video_id"]}_{clip["clip_id"]}'
        if clip_id in clipID2VideoCaption:
            clips_with_caption.append(clip)
        else:
            clips_without_caption.append(clip)
    
    logger.info(f"{srcLanguage} 总clips数量: {len(clips)}")
    logger.info(f"{srcLanguage} 高分clips数量: {len(high_score_clips)}")
    logger.info(f"{srcLanguage} 有caption的clips数量: {len(clips_with_caption)}")
    logger.info(f"{srcLanguage} 没有caption的clips数量: {len(clips_without_caption)}")
    
    # 取前number条有caption的clips
    final_clips = clips_with_caption[:number]
    
    if len(final_clips) < number:
        logger.warning(f"{srcLanguage} 有caption的clips不足{number}条，实际只有{len(final_clips)}条")
    
    return final_clips, clipID2VideoCaption

def makePairedSFTData(clips, clipID2VideoCaption, srcLanguage, tgtLanguage, number):
    """生成两个一一对应的SFT数据集"""
    
    # 数据集1：只有translation
    sftData_translation_only = []
    # 数据集2：有video caption + translation  
    sftData_with_caption = []
    
    for clip in clips:
        clip_id = f'{clip["video_id"]}_{clip["clip_id"]}'
        videoPath = f"./data/TriFine/videoClips/{clip['video_id']}/{clip['video_id']}_{clip['clip_id']}.mp4"
        tgtSent = clip[f"{tgtLanguage.upper()}_sentence"]
        
        # 生成只有translation的数据
        videoTextTranslatePrompt = getUserPrompt(
            promptLanguage="en", 
            srcLanguage=srcLanguage, 
            tgtLanguage=tgtLanguage,
            srcSent=clip[f"{srcLanguage.upper()}_sentence"], 
            dataset_type="video-text"
        )
        
        sftItem_translation = {
            "video": videoPath, 
            "conversations": [ 
                {
                    "from": "human", 
                    "value": f"<video>\n{videoTextTranslatePrompt}"
                },
                {
                    "from": "gpt",
                    "value": tgtSent
                }
            ]
        }
        sftData_translation_only.append(sftItem_translation)
        
        # 生成有video caption + translation的数据
        clipVideoCaption = clipID2VideoCaption[clip_id]
        videoCaptionTranslatePrompt = getUserPrompt(
            promptLanguage="en", 
            srcLanguage=srcLanguage, 
            tgtLanguage=tgtLanguage,
            srcSent=clip[f"{srcLanguage.upper()}_sentence"], 
            dataset_type="video-text", 
            prompt_type="videoCaptionThenTranslate"
        )
        
        sftItem_with_caption = {
            "video": videoPath, 
            "conversations": [ 
                {
                    "from": "human", 
                    "value": f"<video>\n{videoCaptionTranslatePrompt}"
                },
                {
                    "from": "gpt",
                    "value": f"{clipVideoCaption}\n<translation>\n{tgtSent}"
                }
            ]
        }
        sftData_with_caption.append(sftItem_with_caption)
    
    return sftData_translation_only, sftData_with_caption

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number", type=int, default=50000, help="目标数据条数")
    parser.add_argument("--alpha", type=float, default=0.6, help="COMET分数阈值")
    args = parser.parse_args()
    
    fileHandler = logging.FileHandler('./log/cleanTrainData.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    args2Log = "数据构造参数: \n"
    for key, value in vars(args).items():
        args2Log += f"{key}: {value} \n"
    logger.info(args2Log)

    # 检查输出文件是否已存在
    translation_only_file = f"./data/work3/sftData/sftData_translation_only_{args.number}.json"
    with_caption_file = f"./data/work3/sftData/sftData_with_caption_{args.number}.json"
    
    if os.path.exists(translation_only_file) and os.path.exists(with_caption_file):
        logger.info("输出文件已存在，跳过数据生成")
    else:
        # 加载原始数据和分数
        logger.info("加载原始数据和COMET分数...")
        enScores = np.load("./data/work3/dataClean/train_en_comet_scores.npy").tolist()
        zhScores = np.load("./data/work3/dataClean/train_zh_comet_scores.npy").tolist()
        enTrainClips = json.load(open("./data/TriFine/Train_clips_en.json", "r"))
        zhTrainClips = json.load(open("./data/TriFine/Train_clips_zh.json", "r"))
        
        # 过滤并匹配clips
        logger.info("过滤clips并匹配video caption...")
        enClipsFiltered, enClipID2VideoCaption = filterAndMatchClips(
            enTrainClips, enScores, args.alpha, args.number, "en", "zh"
        )
        zhClipsFiltered, zhClipID2VideoCaption = filterAndMatchClips(
            zhTrainClips, zhScores, args.alpha, args.number, "zh", "en"
        )
        
        # 确保两种语言的数据量相同
        min_clips = min(len(enClipsFiltered), len(zhClipsFiltered))
        enClipsFiltered = enClipsFiltered[:min_clips]
        zhClipsFiltered = zhClipsFiltered[:min_clips]
        
        logger.info(f"最终使用的数据量: 英文{len(enClipsFiltered)}条，中文{len(zhClipsFiltered)}条")
        
        # 生成英文到中文的数据
        logger.info("生成英文到中文的SFT数据...")
        en_translation_only, en_with_caption = makePairedSFTData(
            enClipsFiltered, enClipID2VideoCaption, "en", "zh", min_clips
        )
        
        # 生成中文到英文的数据
        logger.info("生成中文到英文的SFT数据...")
        zh_translation_only, zh_with_caption = makePairedSFTData(
            zhClipsFiltered, zhClipID2VideoCaption, "zh", "en", min_clips
        )
        
        # 合并数据并打乱
        logger.info("合并数据并打乱...")
        all_translation_only = en_translation_only + zh_translation_only
        all_with_caption = en_with_caption + zh_with_caption
        
        # 使用相同的随机种子确保两个数据集的顺序一致
        random.seed(42)
        combined = list(zip(all_translation_only, all_with_caption))
        random.shuffle(combined)
        all_translation_only, all_with_caption = zip(*combined)
        all_translation_only = list(all_translation_only)
        all_with_caption = list(all_with_caption)
        
        # 保存数据
        logger.info("保存数据...")
        json.dump(all_translation_only, open(translation_only_file, "w"),
                  ensure_ascii=False, indent=4)
        json.dump(all_with_caption, open(with_caption_file, "w"),
                  ensure_ascii=False, indent=4)
        
        logger.info(f"数据生成完成！")
        logger.info(f"只有translation的数据: {len(all_translation_only)}条")
        logger.info(f"有caption+translation的数据: {len(all_with_caption)}条")
        logger.info(f"保存路径: {translation_only_file}, {with_caption_file}")



