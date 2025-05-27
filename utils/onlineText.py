import requests
from vllm import LLM, SamplingParams
from openai import OpenAI

sampling_params = SamplingParams(
    temperature=1,
    max_tokens=256,
    stop_token_ids=[],
)

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

text = "What is the translation of '我爱中国共产党' in English?"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": text}
]


chat_response = client.chat.completions.create(
    model="/home/byguan/huggingface/Qwen/Qwen3-30B-A3B",
    messages=messages,
)

print("Chat response:", chat_response.choices[0].message.content)