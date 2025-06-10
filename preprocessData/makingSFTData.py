import argparse
import logging
import json
import numpy as np
import random

from utils.prompts import getUserPrompt

logger = logging.getLogger('cleanTrainData')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 

def filterClips(clips, scores, alpha, num, language):
    clipsFiltered = [clip for clip, score in zip(clips, scores) if score > alpha]
    clipsFiltered = clipsFiltered[:num]
    
    logger.info(f"{language} train clips length: {len(clips)}")
    logger.info(f"{language} filtered clips length: {len(clipsFiltered)}")
    logger.info(f"{language} filtered clips ratio: {len(clipsFiltered) / len(clips)}")
    
    json.dump(clipsFiltered, open(f"./data/work3/dataClean/Train_clips_{language}_filtered_{args.num}.json", "w"),\
        ensure_ascii=False, indent=4)
    return clipsFiltered

def makeSFTData(clips, srcLanguage, tgtLanguage, number):
    outputSFTData = []
    
    for clip in clips:
        videoTextTranslatePrompt = getUserPrompt(promptLanguage="en", srcLanguage=srcLanguage, tgtLanguage=tgtLanguage,\
            srcSent=clip[f"{srcLanguage.upper()}_sentence"], dataset_type="video-text")
        videoPath = f"./data/TriFine/videoClips/{clip['video_id']}/{clip['video_id']}_{clip['clip_id']}.mp4"
        SFTItem = { "video": videoPath, 
                    "conversations": [ 
                        {
                            "from": "human", 
                            "value": f"<video>\n{videoTextTranslatePrompt}"
                        },
                        {
                            "from": "gpt",
                            "value": clip[f"{tgtLanguage.upper()}_sentence"]
                        }
                    ]}
        outputSFTData.append(SFTItem)
    json.dump(outputSFTData, open(f"./data/work3/sftData/sftData_{srcLanguage}_{number}.json", "w"),\
        ensure_ascii=False, indent=4)
    return outputSFTData


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=100000)
    parser.add_argument("--alpha", type=float, default=0.6)
    args = parser.parse_args()
    
    
    fileHandler = logging.FileHandler('./log/cleanTrainData.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    
    args2Log = "Data Cleaning arguments: \n"
    for key, value in vars(args).items():
        args2Log += f"{key}: {value} \n"
    logger.info(args2Log)

    enScores = np.load("./data/work3/dataClean/train_en_comet_scores.npy").tolist()
    zhScores = np.load("./data/work3/dataClean/train_zh_comet_scores.npy").tolist()
    enTrainClips = json.load(open("./data/TriFine/Train_clips_en.json", "r"))
    zhTrainClips = json.load(open("./data/TriFine/Train_clips_zh.json", "r"))
    enClipsFiltered = filterClips(enTrainClips, enScores, args.alpha, args.num, "en")
    zhClipsFiltered = filterClips(zhTrainClips, zhScores, args.alpha, args.num, "zh")
    
    sftData_en_zh = makeSFTData(enClipsFiltered, "en", "zh", args.num)
    sftData_zh_en = makeSFTData(zhClipsFiltered, "zh", "en", args.num)
    sftData = sftData_en_zh + sftData_zh_en
    random.shuffle(sftData)
    json.dump(sftData, open(f"./data/work3/sftData/sftData_2_{args.num}.json", "w"),\
        ensure_ascii=False, indent=4)
    
    
    
    
    
    
    
    




