import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import AsyncOpenAI
from qwen_vl_utils import process_vision_info
import asyncio
from tqdm import tqdm
import argparse
import json
from utils.prompts import getUserPrompt
import os
import time
import random
from datetime import datetime
import logging

logger = logging.getLogger('vllmServerInference')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO) 


def prepare_video_message_for_qwen_vllm(content_messages):
    """
    The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
    Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_typ`e of the video explicitly set to `video/jpeg`.
    By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
    """
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                video_message = [{'content': [part_message]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                assert video_inputs is not None, "video_inputs should not be None"
                video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                fps_list.extend(video_kwargs.get('fps', []))

                # encode image with base64
                base64_frames = []
                for frame in video_input:
                    img = Image.fromarray(frame)
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="jpeg")
                    byte_data = output_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    base64_frames.append(base64_str)

                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    return vllm_messages, {'fps': fps_list}


async def process_video_message(args_tuple):
    video_mess, model_path, base_url, clipID = args_tuple
    # 在每个协程中创建 AsyncOpenAI client 实例
    async_process_client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=base_url,
    )
    
    try:
        # print(video_mess)
        video_m, video_kwargs = prepare_video_message_for_qwen_vllm(video_mess)
        # 合并 extra_body 参数
        extra_body_params = {
            "mm_processor_kwargs": video_kwargs,
            "top_k": 20,
        }
        chat_response = await async_process_client.chat.completions.create(
            model=model_path,
            messages=video_m,
            extra_body=extra_body_params,
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
        )
        # print("Chat response:", chat_response.choices[0].message.content)
        return clipID, chat_response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error processing video message: {e}")
        return clipID, f"Error: {str(e)}"
    finally:
        await async_process_client.close()
        
async def process_text_message(args_tuple):
    text_mess, model_path, base_url, clipID = args_tuple
    
    async_process_client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=base_url,
    )
    try:
        # For text messages, we don't have mm_processor_kwargs
        # We also assume text_mess is already in the correct format for the API
        # e.g., [{"role": "user", "content": "some text"}]
        chat_response = await async_process_client.chat.completions.create(
            model=model_path,
            messages=text_mess, # Direct use of text_mess
            max_tokens=8192, # You might want to adjust this for text
            temperature=0.6,
            top_p=0.95,
            extra_body={ # top_k can still be relevant
                "top_k": 20,
            },
        )
        return clipID, chat_response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error processing text message: {e}")
        return clipID, f"Error: {str(e)}"
    finally:
        await async_process_client.close()

# 主函数修改为异步
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filePath", type=str, required=True)
    parser.add_argument("--promptType", type=str, required=True, choices=["textReasoning", "videoCaption"])
    parser.add_argument("--model", type=str, default="/home/byguan/huggingface/Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--dataset_type", type=str, default="video-text", choices=['text', 'video-text', 'image-text'])
    parser.add_argument("-sl", "--srcLanguage", type=str, default="en", choices=['zh', 'en'])
    parser.add_argument("-tl", "--tgtLanguage", type=str, default="zh", choices=['zh', 'en'])
    parser.add_argument("-pl", "--promptLanguage", type=str, default="en", choices=['zh', 'en'])
    parser.add_argument("-s", "--startIndex", type=int, default=None, help="Start index of the data to process, unit 10K")
    parser.add_argument("-e", "--endIndex", type=int, default=None, help="End index of the data to process, unit 10K")
    parser.add_argument("--num_concurrent_requests", type=int, default=10)
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("-p", "--port", type=int, default=8000)
    args = parser.parse_args()
    
    # log 设置 - 创建日期文件夹（与inference.py保持一致）
    while True:
        logDirName = f'./eval/eval-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        if not os.path.exists(logDirName):
            os.makedirs(logDirName)
            break
        else:
            logger.info("Please waiting\n")
            time.sleep(random.randint(1, 10))
            
    fileHandler = logging.FileHandler(f'{logDirName}/eval.log')
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    commandHandler = logging.StreamHandler()
    commandHandler.setLevel(logging.INFO)
    commandHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(commandHandler)
    
    # 将参数记录到日志中
    args2Log = "Script arguments: \n"
    for key, value in vars(args).items():
        args2Log += f"{key}: {value} \n"
    logger.info(args2Log)
    
    vllmURL = f"http://{args.ip}:{args.port}/v1"
    
    with open(args.filePath, "r") as f:
        allData = json.load(f)
    if args.endIndex is not None and args.startIndex is not None:
        allData = allData[args.startIndex*10000:args.endIndex*10000]
    
    semaphore = asyncio.Semaphore(args.num_concurrent_requests)
    clipIDs = [f"{item['video_id']}_{item['clip_id']}" for item in allData]

    all_message_contents = [] # Will store the actual list of messages for the API for each item
    target_process_function = None

    if args.dataset_type == "video-text":
        for clipID in clipIDs:
            prompt = getUserPrompt(args.promptLanguage, args.srcLanguage, args.tgtLanguage, None, 0, args.dataset_type, args.promptType)
            video_message_content_for_api = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "video",
                        "video": f"./data/TriFine/videoClips/{clipID[:11]}/{clipID}.mp4",  
                        "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 2, 
                        'fps': 2.0  # The default value is 2.0
                    }]
                },
            ]
            all_message_contents.append(video_message_content_for_api)
        target_process_function = process_video_message

    elif args.dataset_type == "text":
        for clipData in allData:
            srcSent = clipData[f"{args.srcLanguage.upper()}_sentence"]
            prompt = getUserPrompt(args.promptLanguage, args.srcLanguage, args.tgtLanguage, srcSent, 0, args.dataset_type, args.promptType)
            text_message_content_for_api = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            all_message_contents.append(text_message_content_for_api)
            
        target_process_function = process_text_message
    else:
        logger.error(f"Unsupported dataset_type: {args.dataset_type}. Please use 'video-text' or 'text'. Exiting.")
        return

    if not all_message_contents:
        logger.error("No messages were prepared to process. Exiting.")
        return

    message_args_list = [(content, args.model, vllmURL, clipID) for content, clipID in zip(all_message_contents, clipIDs)]
    
    async def process_item_with_semaphore(arg_tuple_for_processor, actual_processor_func):
        async with semaphore:
            return await actual_processor_func(arg_tuple_for_processor)

    tasks = [process_item_with_semaphore(m_args, target_process_function) for m_args in message_args_list]

    # 直接构建结果字典，按完成顺序收集
    result_dict = {}
    for future in tqdm(asyncio.as_completed(tasks), total=len(message_args_list), desc="Processing"):
        clipID, result = await future
        result_dict[clipID] = result
    
    # 按照原始clipIDs的顺序生成输出
    outputs = []
    for clipID in clipIDs:  # 按原始顺序遍历
        if clipID in result_dict:
            outputs.append({
                "clipID": clipID,
                args.promptType: result_dict[clipID]
            })
        else:
            # 如果某个clipID没有结果（理论上不应该发生），添加错误信息
            outputs.append({
                "clipID": clipID,
                args.promptType: "None"
            })
    
    # 使用新的保存格式（与inference.py保持一致）
    with open(f"{logDirName}/results.json", "w") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)

    logger.info(f"Results saved to {logDirName}/results.json")

if __name__ == '__main__':
    asyncio.run(main())