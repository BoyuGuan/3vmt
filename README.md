# 3vmt
第3个VMT工作



# 数据预处理

## 步骤 1: 提取多模态信息
```bash
python3 ./codes/vllmServerInference.py \
    --filePath ./data/TriFine/Train_clips.json \
    --promptType videoInfoExtraction \
    --model_type multimodal \
    --dataset_type video-text \
    --ip localhost --port 8001
```


## 步骤 2: 处理多模态信息

和SFT Data保持一致
```bash
    python3 ./utils/MMDataAlignWithSFTData.py
```

验证生成的json模板正确性
```bash
python3 ./preprocessData/extractMM.py \
    --MMInfoFilePath ./data/work3/MMinfoAndTrans/results_filtered_by_sft_merged.json \
    --originDataFilePath ./data/TriFine/Train_clips.json \
    --outputFilePath ./data/work3/MMinfoAndTrans/MMInfo.json
```

## 步骤 3: 生成不同的多模态+文本prompt
```bash
python3 ./preprocessData/makeMMPrompt.py \
    --inputFilePath ./data/work3/MMinfoAndTrans/MMInfo.json \
    --outputFilePath ./data/work3/MMinfoAndTrans/data_with_prompts.json \
    --cueTypes all
```

## 步骤 4: 运行翻译实验（不同 cue 类型），构造数据
```bash
python vllm_inference.py \
  --filePath "./data/work3/MMPrompts/data_with_prompts.json" \
  --promptType "mmPromptTranslation" \
  --dataset_type "text" \
  --model_path "./huggingface/Qwen/Qwen3-30B-A3B-Instruct-2507" \
  --mmCueTypes "all" \
  --ip "localhost" \
  --port 8000 \
  --num_concurrent_requests 50
```



# SFT
SFT代码参考的是[Qwen3-VL-官方repo的finetune代码](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)。

<!-- qwenvl/train/argument.py → utils/qwen25VL_sft_argument.py
qwenvl/train/train_qwen.py →  codes/train_qwen25vl_sft.py
qwenvl/train/trainer.py →  codes/qwen25vl_sft_trainer.py

qwenvl/data/\_\_init\_\_.py → vmtDataset/\_\_init\_\_.py
qwenvl/data/data_qwen.py → vmtDataset/data_qwen.py
qwenvl/data/data_qwen_packed.py → vmtDataset/data_qwen_packed.py
qwenvl/data/rope2d.py → vmtDataset/rope2d.py

scripts/sft_7b.sh   →   utils/sft_qwen25VL_7b.sh  
scripts/zero2.json  →   utils/zero2.json
scripts/zero3.json  →   utils/zero3.json -->



