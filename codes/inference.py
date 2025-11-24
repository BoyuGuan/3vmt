import argparse
import tqdm
import random
import time
import av
import numpy as np
import torch
from transformers import AutoModel, AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM, MllamaForConditionalGeneration, LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor 
from torch.utils.data import random_split, DataLoader
from transformers import GenerationConfig
from tqdm import tqdm
import os
import sys
from datetime import datetime
import logging
from utils.prompts import getSystemPrompt, getUserPrompt
from utils.computeTransMetric import computeBLEU, computeMETEOR, computeChrF, computeCOMET, computeBLEURT
import json
from vmt3_dataset.vmtDataset import vmtDatasetForLLM
from qwen_vl_utils import process_vision_info

logger = logging.getLogger('evalModel')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 


# 纯文本大模型跑纯文本数据集
def getSrcPredsRefsTextModel(dataLoader, model, tokenizer, args, generationConfig=None):
    src, preds, refs, clipIDs = [], [], [], []

    # model.eval()
    for batch_data in tqdm(dataLoader):

        src += [w.strip() for w in batch_data["src_text"]]
        refs += [w.strip() for w in batch_data["tgt_text"]]
        clipIDs += [w.strip() for w in batch_data["videoClipID"]]
        
        batchInput = []
        assert len(batch_data["src_text"]) == len(batch_data["tgt_text"]), "Pairs in batch not equal"
        for i in range(len(batch_data["src_text"])):
            inputItem = []
            itemSystemPrompt = getSystemPrompt(args.model_name, args.model_type, args.system_prompt_type)
            if itemSystemPrompt is not None:
                inputItem.append(itemSystemPrompt)
            inputItem.append({"role": "user", "content": getUserPrompt(args.prompt_language, args.source_language, args.target_language,\
                            batch_data["src_text"][i], args.shot_num)})
            batchInput.append(inputItem)
        # print(batchInput)
        # print(batchInput[0])
        text_batch = tokenizer.apply_chat_template(
            batchInput,
            tokenize=False,
            add_generation_prompt=True,
        )
        # model_inputs_batch = tokenizer(text_batch, return_tensors="pt", max_length=args.max_src_length, truncation=True).to(model.device)
        if args.model_name == "Llama-3.1-8B-Instruct":
            tokenizer.pad_token = tokenizer.eos_token
        model_inputs_batch = tokenizer(text_batch, return_tensors="pt", padding=True, max_length=args.max_src_length, truncation=True).to(model.device)
        generated_ids_batch = model.generate(
            **model_inputs_batch,
            max_new_tokens=args.max_tgt_length,
            # generation_config=generationConfig
        )

        generated_ids_batch = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]
        response_batch = tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)
        preds += response_batch
        # for i in range(len(response_batch)):
        #     print(f'Src: {src[i]}')
        #     print(f"Pred: {response_batch[i]}")
        #     print(f"Label: {refs[i]}")
        #     print("=====================================")
        # exit()

    assert len(src) == len(preds) == len(refs), "Lengths of src, preds and refs not equal"
    return src, preds, refs, clipIDs

# 实现了多模态模型跑 纯文本数据集、 视频-文本数据集 和 图片-文本数据集
def getSrcPredsRefsMultimodalModel(dataLoader, model, processor, args, generationConfig=None):
    
    src, preds, refs, clipIDs = [], [], [], []
    tokenizer = processor
    # model.eval()
    for batch_data in tqdm(dataLoader):
        assert len(batch_data["src_text"]) == len(batch_data["tgt_text"]), "Pairs in batch not equal"
        # print("batch_data",batch_data)
        src += [w.strip() for w in batch_data["src_text"]]
        refs += [w.strip() for w in batch_data["tgt_text"]]
        clipIDs += [w.strip() for w in batch_data["videoClipID"]]
        itemSystemPrompt = getSystemPrompt(args.model_name, args.model_type, args.system_prompt_type)
        prompts_raw = []
        prompts = []
        if args.model_name == "InternVideo2_5_Chat_8B":
            pixel_values = batch_data["videoClip"][0]["pixel_values"]
            num_patches_list = batch_data["videoClip"][0]["num_patches_list"]
            pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
            video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
            question1 = getUserPrompt(args.prompt_language, args.source_language, args.target_language, batch_data["src_text"][0], args.shot_num, args.dataset_type, args.prompt_type)
            question = video_prefix + question1
            generation_config = dict(
                do_sample=False,
                temperature=0.0,
                max_new_tokens=1024,
                top_p=0.1,
                num_beams=1
            )
            output_text = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=False)
            preds += [output_text] 
            # print(preds)
            continue  
        elif args.model_name == "InternVL3-14B":
            if args.dataset_type == "video-text":
                pixel_values = batch_data["videoClip"][0]["pixel_values"]
                num_patches_list = batch_data["videoClip"][0]["num_patches_list"]
                pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
                video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
                question1 = getUserPrompt(args.prompt_language, args.source_language, args.target_language, batch_data["src_text"][0], args.shot_num, args.dataset_type, args.prompt_type)
                question = video_prefix + question1
            elif args.dataset_type == "image-text" or args.dataset_type == "images-text":
                # 对于图像输入，需要加载pixel_values
                from utils.InternVideo import load_image
                if args.image_selection == "multiple" or args.image_selection == "chooseImage":
                    # 处理多张图片
                    all_pixel_values = []
                    for img_path in batch_data["imagePath"][0]:
                        pixel_values = load_image(img_path, input_size=448, max_num=12).to(torch.bfloat16).to(model.device)
                        all_pixel_values.append(pixel_values)
                    pixel_values = torch.cat(all_pixel_values, dim=0)
                    image_prefix = "".join([f"<image>\n" for _ in range(len(all_pixel_values))])
                    question1 = getUserPrompt(args.prompt_language, args.source_language, args.target_language, batch_data["src_text"][0], args.shot_num, args.dataset_type, args.prompt_type)
                    question = image_prefix + question1
                else:
                    # 处理单张图片
                    pixel_values = load_image(batch_data["imagePath"][0], input_size=448, max_num=12).to(torch.bfloat16).to(model.device)
                    question1 = getUserPrompt(args.prompt_language, args.source_language, args.target_language, batch_data["src_text"][0], args.shot_num, args.dataset_type, args.prompt_type)
                    question = "<image>\n" + question1
            else:
                # 纯文本
                question = getUserPrompt(args.prompt_language, args.source_language, args.target_language, batch_data["src_text"][0], args.shot_num, args.dataset_type, args.prompt_type)
                pixel_values = None
            
            generation_config = dict(
                max_new_tokens=args.max_tgt_length, 
                do_sample=True,
                temperature=0.1,
                top_p=0.9
            )
            if pixel_values is not None:
                output_text = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=False)
            else:
                # 纯文本情况
                output_text, _ = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
            preds += [output_text] 
            continue
        for i in range(len(batch_data["src_text"])):
            promptItem = []
            if itemSystemPrompt is not None:
                promptItem.append(itemSystemPrompt)
            userPrompt = getUserPrompt(args.prompt_language, args.source_language, args.target_language, batch_data["src_text"][i], args.shot_num, args.dataset_type, args.prompt_type)
            if args.dataset_type == "text":
                if args.model_name == "MiniCPM-V-2_6":
                    promptItem.append({"role": "user", "content": [userPrompt]})
                else:
                    promptItem.append({"role": "user", "content": [{"type": "text", "text": userPrompt}]})
            elif args.dataset_type == "video-text":
                assert len(batch_data["tgt_text"]) == len(batch_data["videoClip"]), "Video-text pairs in batch not equal"
                if args.model_name == "LLaVA-NeXT-Video-7B-hf":
                    promptItem.append({"role": "user", "content": [{"type": "text", "text": userPrompt}, {"type": "video"}]})
                elif "Qwen2-VL" in args.model_name or "Qwen2.5-VL" in args.model_name or "Qwen3-VL" in args.model_name:
                    video_path = batch_data["videoClipPath"][i]
                    promptItem.append({"role": "user", "content": [{"type": "video","video": video_path, "max_pixels": 360 * 420, "fps": 1.0,}, {"type": "text", "text": userPrompt}]})
                    generationConfig = None
                elif args.model_name == "MiniCPM-V-2_6":
                    promptItem.append({"role": "user", "content": batch_data["videoClip"][i] + [userPrompt]})
                else:
                    raise TypeError("Model name error!")
            elif args.dataset_type == "image-text":
                assert len(batch_data["src_text"]) == len(batch_data["imagePath"]), "Image-text pairs in batch not equal"
                if "Qwen2-VL" in args.model_name or "Qwen2.5-VL" in args.model_name or "Qwen3-VL" in args.model_name :
                    image_path = batch_data["imagePath"][i]
                    if args.image_selection == "multiple":
                        contentList = [{"type": "image", "image": path, "max_pixels": 360 * 420} for path in image_path]
                    else:
                        contentList = [{"type": "image", "image": image_path, "max_pixels": 360 * 420}]
                    contentList += [{"type": "text", "text": userPrompt}]
                    promptItem.append({"role": "user", "content": contentList})
                    generationConfig = None
                elif args.model_name == "LLaVA-NeXT-Video-7B-hf":
                    promptItem.append({"role": "user", "content": [{"type": "text", "text": userPrompt}, {"type": "image"},]})
                elif args.model_name == "MiniCPM-V-2_6":
                    promptItem.append({"role": "user", "content": [batch_data["image"][i]] + [userPrompt]})
                elif args.model_name == "Llama-3.2-11B-Vision-Instruct":
                    promptItem.append({"role": "user", "content": [{"type": "image"},{"type": "text", "text": userPrompt}]})
                else:
                    raise TypeError("Model name error!")
            else:
                raise TypeError("Dataset type error!")
            
            prompts_raw.append(promptItem)   
            
            if args.model_name != "MiniCPM-V-2_6":
                # 默认tokenize=False
                prompt = processor.apply_chat_template(promptItem, add_generation_prompt=True)
                prompts.append(prompt)

        if args.model_name == "MiniCPM-V-2_6":
            # 根据文档，image=None 需要显示设定
            output_text = model.chat(
                image=None,
                msgs=prompts_raw,
                tokenizer=tokenizer
            )
        else:
            if args.dataset_type == "text":
                inputs_batch = processor(text=prompts, padding=True, return_tensors="pt").to(model.device)
            elif args.dataset_type == "video-text":
                videos = batch_data["videoClip"]
                if "Qwen2-VL" in args.model_name or "Qwen2.5-VL" in args.model_name or "Qwen3-VL" in args.model_name:
                    images, videos = process_vision_info(prompts_raw)
                inputs_batch = processor(text=prompts, images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
            elif args.dataset_type == "image-text" or args.dataset_type == "images-text":
                images = batch_data["image"]
                if "Qwen2-VL" in args.model_name or "Qwen2.5-VL" in args.model_name or "Qwen3-VL" in args.model_name:
                    images, videos = process_vision_info(prompts_raw)
                if args.model_name == "Llama-3.2-11B-Vision-Instruct":
                    inputs_batch = processor(images=images,text=prompts,add_special_tokens=False,padding=True, return_tensors="pt").to(model.device)
                else:
                    inputs_batch = processor(text=prompts, images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
            else :
                raise TypeError("Dataset type format error!")

            output_ids = model.generate(
                **inputs_batch,
                max_new_tokens=args.max_tgt_length,
                # generation_config=generationConfig
            )
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs_batch.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # print(output_text)
        preds += output_text
        # outputs = processor.batch_decode(generated_ids_batch, skip_special_tokens=True)
        # print("outputs")
        # print(outputs)
        # results = []
        # for outputItem in outputs:
        #     index = outputItem.find('ASSISTANT:')
        #     if index == -1:
        #         print("error")
        #         exit()
        #     results.append(outputItem[index + len('ASSISTANT: '):])
        # print("results")
        # print(results)
        # preds += results

    # print("preds")
    # print(preds)
    return src, preds, refs, clipIDs

# 处理纯文本数据集
def text_collate_fn(batchRawData):
    batchData = {"src_text":[], "tgt_text":[], "videoClipID":[]}
    for dataItem in batchRawData:
        batchData["src_text"].append(dataItem['src_text'])
        batchData["tgt_text"].append(dataItem['tgt_text'])
        batchData["videoClipID"].append(dataItem['videoClipID'])
    return batchData

# 处理视频-文本数据集
def video_text_collate_fn(batchRawData):
    batchData = {"src_text":[], "tgt_text":[], "videoClipID":[], "videoClip":[], "videoClipPath":[]}
    for dataItem in batchRawData:
        batchData["src_text"].append(dataItem['src_text'])
        batchData["tgt_text"].append(dataItem['tgt_text'])
        batchData["videoClipID"].append(dataItem['videoClipID'])
        batchData["videoClip"].append(dataItem['videoClip'])
        batchData["videoClipPath"].append(dataItem['videoClipPath'])
    return batchData

# 处理图片-文本数据集
def image_text_collate_fn(batchRawData):
    batchData = {"src_text":[], "tgt_text":[],"imagePath":[],"image":[],"videoClipID":[]}
    for dataItem in batchRawData:
        batchData["src_text"].append(dataItem['src_text'])
        batchData["tgt_text"].append(dataItem['tgt_text'])
        batchData["imagePath"].append(dataItem['imagePath'])
        batchData["image"].append(dataItem['image'])
        batchData["videoClipID"].append(dataItem['videoClipID'])
    return batchData

# 保存翻译结果，方便查看
def saveResult(src, preds, refs, clipIDs, logDirName):
    assert len(src) == len(preds) == len(refs) == len(clipIDs), "Lengths of src, preds, refs and clipIDs not equal"
    data_list = []

    for srcSent, predSent, refSent, clipID in zip(src, preds, refs, clipIDs):
        data = {
            "clipID": clipID,
            "src": srcSent,
            "preds": predSent,
            "refs": refSent
        }
        data_list.append(data)

    with open(f"{logDirName}/results.json", 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, default=None)
    parser.add_argument("--dataset_type", type=str, default='text', help='Dataset type (text, video-text, image-text)')
    parser.add_argument("--model_type", type=str, default='multimodal', help='Model type (text or multimodal), default is multimodal')
    parser.add_argument("--video_path", type=str, default='./data/TriFine/videoClips', help='Video path')
    parser.add_argument("--model_path", type=str, default='./huggingface/Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--image_selection", type=str, default=None, help='Image selection (mid, random, multiple, select, given)')
    parser.add_argument('-sl', "--source_language", type=str, default='en', help='Source language (zh or en)')
    parser.add_argument('-tl', "--target_language", type=str, default='zh', help='Target language (zh or en)')
    parser.add_argument('-pl', "--prompt_language", type=str, default='en', help='Prompt language (zh or en)')
    parser.add_argument('-spt', '--system_prompt_type', type=str, default='default')    
    parser.add_argument('-sn', '--shot_num', type=int, default=0)    
    # parser.add_argument('--is_test_set', action='store_true', help="Whether to evaluate test set.")
    parser.add_argument('-bs', '--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument("--max_src_length", type=int, default= 1024)
    parser.add_argument("--max_tgt_length", type=int, default= 256)
    # parser.add_argument("--generation_config_dir", type=str, default='./checkpoint/config/generationConfig')
    parser.add_argument("--generation_config_dir", type=str, default=None)

    parser.add_argument("--trans_metric", action="store_false", help="Whether to compute translation metrics. Default is True, if set it is False.")
    parser.add_argument("--metrics", nargs='+', default=['BLEU', 'METEOR', 'chrF', 'COMET', 'BLEURT'],
                    help="Specify which metrics to compute. Available options: BLEU, METEOR, chrF, COMET, BLEURT. "
                            "Default is to compute all metrics.")
    parser.add_argument("-pt", "--prompt_type", type=str, default=None, help="Prompt type, default is None, which means of translation prompt. And can set to `chooseImage` to choose image.")
    parser.add_argument("-s", "--special", type=str, default=None, help="Special setting")
    parser.add_argument("--cluster_path", type=str, default=None, help="Image path")
    parser.add_argument("--picID_path", type=str, default=None, help="The clip ID to picture ID file path.")
    parser.add_argument("--given_pic_ID", type=int, default= None)
    parser.add_argument("--vatex", action="store_true", help="Whether to use VATEX dataset.")
    
    args = parser.parse_args()

    # log 设置
    while True:
        logDirName = f'./eval/eval-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        if not os.path.exists(logDirName):
            os.makedirs(logDirName)
            break
        else:
            print("Please waiting\n")
            time.sleep(random.randint(60, 240))
            
    fileHandler = logging.FileHandler(f'{logDirName}/eval.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    if args.model_name is None:
        args.model_name = args.model_path.split('/')[-1]
    
    if args.model_name == "InternVideo2_5_Chat_8B":
        args.batch_size = 1
        print("InternVideo2_5_Chat_8B model only support batch_size=1")
        
    if args.model_name == "InternVL3-14B":
        args.batch_size = 1
        print("InternVL3-14B model only support batch_size=1")
        
    # 将参数记录到日志中
    args2Log = "Script arguments: \n"
    for key, value in vars(args).items():
        args2Log += f"{key}: {value} \n"
    logger.info(args2Log)

    allDataset = vmtDatasetForLLM(args)
    if args.dataset_type == "text":         # 文本数据集
        modelName2collate_fn = text_collate_fn
    elif args.dataset_type == "video-text": # 视频-文本数据集
        modelName2collate_fn = video_text_collate_fn
    elif args.dataset_type == "image-text" or args.dataset_type == "images-text": # 图片-文本数据集
        modelName2collate_fn = image_text_collate_fn
    else:
        raise TypeError("Dataset type format error!")
    
    # if args.is_test_set:   默认为全推理
    testDataloader = DataLoader(allDataset, collate_fn=modelName2collate_fn, batch_size=args.batch_size,\
        num_workers=16, pin_memory=True, shuffle=False, prefetch_factor=2, persistent_workers=True)
    # else:
    #     datasetGenerator = torch.Generator().manual_seed(42)
    #     trainDataset, validDataset = random_split(allDataset, [0.95, 0.05], generator=datasetGenerator)
    #     testDataloader = DataLoader(validDataset, collate_fn=modelName2collate_fn[args.model_name], batch_size=args.batch_size,\
    #         num_workers=16, pin_memory=False, shuffle=False, prefetch_factor=2)
    
    if args.generation_config_dir is not None:
        generationConfig = GenerationConfig.from_pretrained(args.generation_config_dir)
    else:
        generationConfig = None
    
    # 对纯文本和多模态大模型做了区分
    if args.model_type == "text":
        if "Qwen2" in args.model_name or "Qwen3" in args.model_name or "Llama-3" in args.model_name:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
            model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto")
        elif args.model_name == "internlm3-8b-instruct":
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
            model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
            model.eval()
        else:
            raise TypeError("Model name not supported!")
        src, preds, refs, clipIDs = getSrcPredsRefsTextModel(testDataloader, model, tokenizer, args, generationConfig)
        # src, preds, refs = getSrcPredsRefs(testDataloader, model, tokenizer, model.generation_config)
    elif args.model_type == "multimodal":
        if args.model_name == "LLaVA-NeXT-Video-7B-hf":
            model = LlavaNextVideoForConditionalGeneration.from_pretrained(args.model_path, torch_dtype="auto",device_map="auto")
            processor = LlavaNextVideoProcessor.from_pretrained(args.model_path)
        elif "Qwen2-VL" in args.model_name:
            model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype="auto",device_map="auto")
            processor = AutoProcessor.from_pretrained(args.model_path)
        elif "Qwen2.5-VL" in args.model_name:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype="auto",device_map="auto")
            processor = AutoProcessor.from_pretrained(args.model_path, padding_side='left')
        elif  "Qwen3-VL" in args.model_name:
            model = AutoModelForImageTextToText.from_pretrained(args.model_path, torch_dtype="auto",device_map="auto")
            processor = AutoProcessor.from_pretrained(args.model_path, padding_side='left')

        elif args.model_name == "MiniCPM-V-2_6":
            # 使用torch_dtype="auto"会报错
            model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
                attn_implementation='sdpa', 
                ) # sdpa or flash_attention_2, no eager
            model = model.eval().cuda()
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            # 这样写方便下面传参
            processor = tokenizer
        elif args.model_name == "InternVideo2_5_Chat_8B":
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(args.model_path,trust_remote_code=True).half().cuda().to(torch.bfloat16)
            processor = tokenizer
        elif args.model_name == "InternVL3-14B":
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
            # 参考2.py中的模型加载方式
            model = AutoModel.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map="auto").eval()
            processor = tokenizer
        elif args.model_name == "Llama-3.2-11B-Vision-Instruct":
            model = MllamaForConditionalGeneration.from_pretrained(args.model_path,torch_dtype=torch.bfloat16,device_map="auto")
            processor = AutoProcessor.from_pretrained(args.model_path, padding_side='left')
        else:
            raise TypeError("Model name not supported!")
        src, preds, refs, clipIDs = getSrcPredsRefsMultimodalModel(testDataloader, model, processor, args, generationConfig)
    else:
        raise TypeError("Model type format error!")
    saveResult(src, preds, refs, clipIDs, logDirName)

    if args.trans_metric:
        # 计算翻译指标
        logger.info(f"\033[91m Evaluation Resutlts")
        if 'BLEU' in args.metrics:
            logger.info(f"\033[91m BLEU: {computeBLEU(preds, refs, args.target_language == 'zh', usingSacreBLEU=False)}   ( Using Huggingface SacreBLEU )  \033[0m")
            logger.info(f"\033[91m BLEU: {computeBLEU(preds, refs, args.target_language == 'zh', usingSacreBLEU=True)}    ( Using SacreBLEU )   \033[0m")
        if 'METEOR' in args.metrics:
            logger.info(f"\033[91m METEOR: {computeMETEOR(preds, refs, args.target_language == 'zh') * 100} \033[0m")
        if 'chrF' in args.metrics:
            logger.info(f'\033[91m chrF: {computeChrF(preds, refs)} \033[0m')
        if 'COMET' in args.metrics:
            logger.info(f'\033[91m COMET: {computeCOMET(src, preds, refs)["mean_score"] * 100} \033[0m')
        if 'BLEURT' in args.metrics:
            logger.info(f'\033[91m BLEURT: {computeBLEURT(preds, refs) * 100} \033[0m')