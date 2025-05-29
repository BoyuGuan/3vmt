import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI
from qwen_vl_utils import process_vision_info
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import json
from utils.getPrompts import get_prompts

def prepare_message_for_vllm(content_messages):
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


def process_video_message(args_tuple):
    video_mess, model_path = args_tuple
    # 在每个进程中创建client实例，确保多进程安全
    process_client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )
    
    try:
        # print(video_mess)
        video_m, video_kwargs = prepare_message_for_vllm(video_mess)
        chat_response = process_client.chat.completions.create(
            model=model_path,
            messages=video_m,
            extra_body={
                "mm_processor_kwargs": video_kwargs
            }
        )
        # print("Chat response:", chat_response.choices[0].message.content)
        return chat_response.choices[0].message.content
    except Exception as e:
        print(f"Error processing video message: {e}")
        return f"Error: {str(e)}"

# 多进程处理 - 使用multiprocessing.Pool和imap
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filePath", type=str, required=True)
    parser.add_argument("--model", type=str, default="/home/byguan/huggingface/Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--num_processes", type=int, default=4)
    args = parser.parse_args()
    
    with open(args.filePath, "r") as f:
        data = json.load(f)
    clipIDs = [f"{item['video_id']}_{item['clip_id']}" for item in data]
    
    video_messages = []
    for clipID in clipIDs:
        prompt = get_prompts("videoCaption", "zh" if "zh" in args.filePath.lower().split("/")[-1] else "en")
        video_messages.append(
            [
                {"role": "system", "content": "You are a helpful assistant."},
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
        )

    # # 为每个video_message添加model参数
    video_message_args = [(msg, args.model) for msg in video_messages]
    
    with Pool(processes=args.num_processes) as p:
        results = list(tqdm(p.imap(process_video_message, video_message_args), total=len(video_messages)))
    outputs = []
    for clipID, result in zip(clipIDs, results):
        outputs.append({
            "clipID": clipID,
            "videoCaption": result
        })
    
    with open(f"./data/work3/videoCaption/{args.filePath.split('/')[-1].split('.')[0]}.json", "w") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)