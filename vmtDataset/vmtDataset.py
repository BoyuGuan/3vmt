import json
import av
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import io
from typing import Dict
import random
from PIL import Image
from decord import VideoReader, cpu    # pip install decord
from utils.InternVideo import load_video,load_image


class Dataset4Clip(Dataset):
    def __init__(self, args):
        with open(args.clustered_file_path, 'r') as f:
            self.clustred_data = json.load(f)
        with open(args.test_file_path, 'r') as f:
            self.test_data = json.load(f)
        self.src_language = args.src_language
        self.frams_path = args.frams_path
        assert len(self.clustred_data) == len(self.test_data), "Length of data not equal"

    def __len__(self):
        return len(self.clustred_data)
    
    def __getitem__(self, idx):
        itemData = {}
        assert list(self.clustred_data[idx].keys())[0] == self.test_data[idx]['video_id']+"_"+str(self.test_data[idx]['clip_id']), "video_id or clip_id not equal"
        itemData['video_id'] = self.test_data[idx]['video_id']
        itemData['clip_id'] = self.test_data[idx]['clip_id']
        itemData['videoClipID'] = itemData["video_id"]+"_"+str(itemData["clip_id"])
        itemData['cluster_ids'] = self.clustred_data[idx][list(self.clustred_data[idx].keys())[0]]
        
        if self.src_language == "en":
            itemData['text'] = self.test_data[idx]['EN_sentence']
        elif self.src_language == "zh":
            itemData['text'] = self.test_data[idx]['ZH_sentence']
        else:
            raise Exception("language not support")
        
        itemData['images'] = []
        for j in range(len(itemData['cluster_ids'])):
            image_path = os.path.join(self.frams_path, itemData['video_id'], itemData['video_id'] + "_" + str(itemData['clip_id']) + "_" + str(itemData['cluster_ids'][j]) + ".jpg")
            image = Image.open(image_path)
            itemData['images'].append(image)
        return itemData

class vmtDatasetForLLM(Dataset):
    """
        多模态数据集，返回句对和视频
    """
    def __init__(self, args):
        with open(args.dataset_path, 'r') as f:
            clipData = json.load(f)
        self.clipData = clipData
        self.modelName = args.model_name
        self.srcLanguage = args.source_language
        self.tgtLanguage = args.target_language
        self.promptLanguage = args.prompt_language
        self.allVideoPath = args.video_path
        self.datasetType = args.dataset_type
        self.imageSelection = args.image_selection
        self.special = args.special
        self.vatex = args.vatex
        
        if args.image_selection == "select":
            assert args.cluster_path is not None, "Please provide the cluster path!"
            assert args.picID_path is not None, "Please provide the picID path!"
            with open(args.cluster_path, 'r') as f:
                clusterInfo = json.load(f)
            clusterData = {k: v for clipCluster in clusterInfo for k, v in clipCluster.items()}
            with open(args.picID_path, 'r') as f:
                picIDChoose = json.load(f)
            clipPicIDChoose = {info["clipID"]: info["picIDChoose"] for info in picIDChoose}
            self.clip2PicID = {key: clusterData[key][value] for key, value in clipPicIDChoose.items()}
        
        if args.image_selection == "given":
            assert args.cluster_path is not None, "Please provide the cluster path!"
            assert args.given_pic_ID is not None, "Please provide the given pic ID!"
            with open(args.cluster_path, 'r') as f:
                clusterInfo = json.load(f)
            self.clip2PicID = {k: v[args.given_pic_ID] for clipCluster in clusterInfo for k, v in clipCluster.items()}
        
    def __len__(self):
        return len(self.clipData)
    
    # https://huggingface.co/docs/transformers/main/en/model_doc/qwen2_vl
    # Qwen2-VL-7B-Instruct处理视频的函数
    # def _fetch_video(self, ele: Dict, nframe_factor=2):
    #     if isinstance(ele['video'], str):
    #         def round_by_factor(number: int, factor: int) -> int:
    #             return round(number / factor) * factor

    #         video = ele["video"]
    #         if video.startswith("file://"):
    #             video = video[7:]

    #         video, _, info = io.read_video(
    #             video,
    #             start_pts=ele.get("video_start", 0.0),
    #             end_pts=ele.get("video_end", None),
    #             pts_unit="sec",
    #             output_format="TCHW",
    #         )
    #         assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    #         if "nframes" in ele:
    #             nframes = round_by_factor(ele["nframes"], nframe_factor)
    #         else:
    #             fps = ele.get("fps", 1.0)
    #             nframes = round_by_factor(video.size(0) / info["video_fps"] * fps, nframe_factor)
    #         idx = torch.linspace(0, video.size(0) - 1, nframes, dtype=torch.int64)
    #         return video[idx]

    # https://huggingface.co/openbmb/MiniCPM-V-2_6
    # MiniCPM-V-2_6处理视频的函数
    def _encode_video(self,video_path):
        MAX_NUM_FRAMES=64 # if cuda OOM set a smaller number
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > MAX_NUM_FRAMES:
            frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        # print('num frames:', len(frames))
        return frames
    
    # https://huggingface.co/docs/transformers/main/en/model_doc/llava_next_video
    # LLaVA-NeXT-Video-7B-hf处理视频的函数
    def _read_video_pyav(self, container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def _get_video_from_path(self, videoPath, modelName):
        if modelName == "LLaVA-NeXT-Video-7B-hf":
            container = av.open(videoPath)
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / 8).astype(int)
            return self._read_video_pyav(container, indices)
        elif "Qwen2-VL" in modelName or "Qwen2.5-VL" in modelName:

            # video_info = {"type": "video", "video": videoPath, "fps": 0.2}
            # video = self._fetch_video(video_info)
            return None
        elif modelName == "MiniCPM-V-2_6":
            return self._encode_video(videoPath)
        elif modelName == "InternVideo2_5_Chat_8B":
            num_segments=128
            pixel_values, num_patches_list = load_video(videoPath, num_segments=num_segments, max_num=1, get_frame_by_duration=False)
            return {"pixel_values":pixel_values, "num_patches_list":num_patches_list}
        else:
            raise TypeError("Model name error!")
        
    def _get_image_from_path(self, imagePath, modelName):
        if modelName == "LLaVA-NeXT-Video-7B-hf":
            return Image.open(imagePath)
        elif "Qwen2-VL" in modelName or "Qwen2.5-VL" in modelName:
            return None
        elif modelName == "MiniCPM-V-2_6":
            if isinstance(imagePath, list):
                return [ Image.open(singlePath).convert('RGB') for singlePath in imagePath]
            return Image.open(imagePath).convert('RGB')
        elif modelName == "Llama-3.2-11B-Vision-Instruct":
            return [Image.open(imagePath)]
        else:
            raise TypeError("Model name error!")

    def __getitem__(self, idx):
        thisItem = dict()
        originClipData = self.clipData[idx]
        thisItem["src_text"] = originClipData[f"{self.srcLanguage.upper()}_sentence"]
        thisItem["tgt_text"] = originClipData[f"{self.tgtLanguage.upper()}_sentence"]
        if self.vatex:
            thisItem["videoClipID"] = f"{originClipData['videoClipId']}"
        else:
            thisItem["videoClipID"] = f"{originClipData['video_id']}_{originClipData['clip_id']}"
        if self.datasetType == "video-text":
            
            if self.vatex:
                videoClipPath = os.path.join(self.allVideoPath, originClipData['videoClipId'] + ".mp4")
            else:
                videoClipPath = os.path.join(self.allVideoPath, originClipData['video_id'], f"{thisItem['videoClipID']}.mp4")
            thisItem["videoClipPath"] = videoClipPath
            thisItem["videoClip"] = self._get_video_from_path(videoClipPath, self.modelName)
            thisItem["startSecond"] = max(0, originClipData["subtitle_start_second"] - originClipData["clip_start_second"])
            thisItem["endSecond"] = min(10, originClipData["subtitle_end_second"] - originClipData["clip_start_second"])
        elif self.datasetType == "image-text":
            if self.imageSelection == "mid":
                if self.vatex:
                    imagePath = f'./data/VATEX_ZH/50frames/{originClipData["videoClipId"]}/{thisItem["videoClipID"]}_mid.jpg'
                else:
                    imagePath = f'./data/work2/frames/10frames/{originClipData["video_id"]}/{thisItem["videoClipID"]}_mid.jpg'
            elif self.imageSelection == "random":
                number = random.randint(0,49)
                if self.vatex:
                    imagePath = f'./data/VATEX_ZH/50frames/{originClipData["videoClipId"]}/{thisItem["videoClipID"]}_{number}.jpg'
                else:
                    imagePath = f'./data/work2/frames/50frames/{originClipData["video_id"]}/{thisItem["videoClipID"]}_{number}.jpg'
            elif self.imageSelection == "multiple":
                if self.vatex:
                    imagePath = [f'./data/VATEX_ZH/50frames/{originClipData["videoClipId"]}/{thisItem["videoClipID"]}_{i*5}.jpg' for i in range(10)]
                elif self.special == "changePicID": # 用于测试在改变picID的情况下，模型推理的图片是否具有一致性
                    imagePath = [f'./data/work2/frames/10frames/{originClipData["video_id"]}/{thisItem["videoClipID"]}_{(i+5)%10}.jpg' for i in range(10)]
                else:
                    imagePath = [f'./data/work2/frames/10frames/{originClipData["video_id"]}/{thisItem["videoClipID"]}_{i}.jpg' for i in range(10)]                    
            elif self.imageSelection == "select" or self.imageSelection == "given":
                if self.vatex:
                    imagePath =  f'./data/VATEX_ZH/50frames/{originClipData["videoClipId"]}/{thisItem["videoClipID"]}_{self.clip2PicID[thisItem["videoClipID"]]}.jpg'
                else:
                    imagePath =  f'./data/work2/frames/50frames/{originClipData["video_id"]}/{thisItem["videoClipID"]}_{self.clip2PicID[thisItem["videoClipID"]]}.jpg'
            else:
                raise ValueError("Image selection error!")
            thisItem["imagePath"] = imagePath
            thisItem["image"] = self._get_image_from_path(imagePath, self.modelName)
        elif self.datasetType == "text": 
            pass
        else:
            raise TypeError("Dataset type error!")
            
        return thisItem


if __name__ == '__main__':
    mySet = vmtDatasetForLLM("/home/byguan/LLMvmt/data/legal_premix_en_clips.json")
    loa = DataLoader(mySet, batch_size=4)
    for s in loa:
        print(s)
        print(type(s))
        exit()
    
    # mySet = Dataset4SelectorTrain("./data/work2/train_en_0_20_filtered.json")
    # loa = DataLoader(mySet, batch_size=4)
    # for s in loa:
    #     print(s)
    #     print(type(s))
    #     exit()