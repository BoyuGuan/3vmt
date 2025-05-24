from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os

MODEL_PATH = "/home/byguan/huggingface/Qwen/Qwen2.5-VL-32B-Instruct"

# 方法1：使用环境变量指定特定的GPU设备（例如GPU 0和1）
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# 如果想使用其他GPU组合，可以修改为：
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # 使用GPU 1和2
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"  # 使用GPU 0和3
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # 使用GPU 2和3

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
    tensor_parallel_size=2,  # 使用2个GPU进行tensor并行
    gpu_memory_utilization=0.9,  # GPU内存使用率
    # 可选的其他参数：
    # max_model_len=32768,         # 最大模型长度
    # enforce_eager=True,          # 强制使用eager模式，有时对多GPU更稳定
)

sampling_params = SamplingParams(
    temperature=1,
    max_tokens=256,
    stop_token_ids=[],
)

image_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png",
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28,
            },
            {"type": "text", "text": "What is the text in the illustrate?"},
        ],
    },
]


# For video input, you can pass following values instead:
# "type": "video",
# "video": "<video URL>",
video_messages1 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
            {"type": "text", "text": "描述一下这个视频"},
            {
                "type": "video", 
                "video": "/home/byguan/LLMvmt/temp/3.mp4",
                "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28
            }
        ]
    },
]

video_messages2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
            {"type": "text", "text": "这个视频中有几个人？"},
            {
                "type": "video", 
                "video": "/home/byguan/LLMvmt/temp/3.mp4",
                "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28
            }
        ]
    },
]

video_messages3 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
            {"type": "text", "text": "这个视频发生在哪？"},
            {
                "type": "video", 
                "video": "/home/byguan/LLMvmt/temp/3.mp4",
                "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28
            }
        ]
    },
]


# vllm会自动进行batch处理，无需手动划分batch
video_messages_list = [video_messages1, video_messages2, video_messages3]*50

# Here we use video messages as a demonstration
inputs = []
for video_messages in video_messages_list:

    messages = video_messages

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,

        # FPS will be returned in video_kwargs
        "mm_processor_kwargs": video_kwargs,
    }
    
    inputs.append(llm_inputs)

outputs = llm.generate(inputs, sampling_params=sampling_params)
for output in outputs:
    generated_text = output.outputs[0].text
    print(generated_text)
    print("\n--------------------------------\n")