import json
from transformers import AutoTokenizer

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained('./huggingface/Qwen/Qwen2.5-VL-7B-Instruct')

# 加载两种数据
video_caption_data = json.load(open('/home/byguan/3vmt/data/work3/sftData/videoCaption-SFTData_2_50000.json'))
direct_data = json.load(open('/home/byguan/3vmt/data/work3/sftData/sftData_2_50000.json'))

# 统计长度
def get_output_length(data_sample):
    assistant_message = data_sample['conversations'][1]['value']
    tokens = tokenizer.encode(assistant_message)
    return len(tokens)

def get_total_length(data_sample):
    # 计算完整对话的长度
    conversation_text = ""
    for conv in data_sample['conversations']:
        conversation_text += conv['value'] + "\n"
    tokens = tokenizer.encode(conversation_text)
    return len(tokens)

# 计算前100个样本的统计
video_caption_output_lengths = [get_output_length(sample) for sample in video_caption_data[:100]]
direct_output_lengths = [get_output_length(sample) for sample in direct_data[:100]]

video_caption_total_lengths = [get_total_length(sample) for sample in video_caption_data[:100]]
direct_total_lengths = [get_total_length(sample) for sample in direct_data[:100]]

print("=== 输出序列长度统计 ===")
print(f'Video Caption + Translation 平均输出长度: {sum(video_caption_output_lengths)/len(video_caption_output_lengths):.1f} tokens')
print(f'Direct Translation 平均输出长度: {sum(direct_output_lengths)/len(direct_output_lengths):.1f} tokens')
print(f'输出长度比例: {(sum(video_caption_output_lengths)/len(video_caption_output_lengths)) / (sum(direct_output_lengths)/len(direct_output_lengths)):.1f}x')

print("\n=== 总序列长度统计 ===")
print(f'Video Caption + Translation 平均总长度: {sum(video_caption_total_lengths)/len(video_caption_total_lengths):.1f} tokens')
print(f'Direct Translation 平均总长度: {sum(direct_total_lengths)/len(direct_total_lengths):.1f} tokens')
print(f'总长度比例: {(sum(video_caption_total_lengths)/len(video_caption_total_lengths)) / (sum(direct_total_lengths)/len(direct_total_lengths)):.1f}x')

print("\n=== 长度分布 ===")
print(f'Video Caption + Translation 输出长度范围: {min(video_caption_output_lengths)} - {max(video_caption_output_lengths)}')
print(f'Direct Translation 输出长度范围: {min(direct_output_lengths)} - {max(direct_output_lengths)}')