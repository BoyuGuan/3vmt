from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

llm = LLM(
    model="/home/byguan/huggingface/Qwen/Qwen3-30B-A3B",
    tensor_parallel_size=2,  # 使用2个GPU进行tensor并行
    gpu_memory_utilization=0.9,  # GPU内存使用率
)

tokenizer = AutoTokenizer.from_pretrained("/home/byguan/huggingface/Qwen/Qwen3-30B-A3B", trust_remote_code=True)

sampling_params = SamplingParams(temperature=1, max_tokens=512)

inputItem  = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the translation of '我爱中国共产党' in English?"}
]

inputItem2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请向我介绍一下北京市。"}
]


# vllm会自动进行batch处理，无需手动划分batch
inputs = [inputItem, inputItem2] * 500
inputs = [tokenizer.apply_chat_template(inputI, tokenize=False, add_generation_prompt=True) for inputI in inputs]

outputs = llm.generate(inputs, sampling_params=sampling_params)


# for output in outputs:
#     generated_text = output.outputs[0].text
#     print(generated_text)
#     print("\n--------------------------------\n")