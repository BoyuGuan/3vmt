# 3vmt
第3个VMT工作





# SFT
SFT代码参考的是[Qwen2.5-VL-官方repo的finetune代码](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune)。


qwenvl/train/argument.py → utils/qwen25VL_sft_argument.py
qwenvl/train/train_qwen.py →  codes/train_qwen25vl_sft.py
qwenvl/train/trainer.py →  codes/qwen25vl_sft_trainer.py

qwenvl/data/\_\_init\_\_.py → vmtDataset/\_\_init\_\_.py
qwenvl/data/data_qwen.py → vmtDataset/data_qwen.py
qwenvl/data/data_qwen_packed.py → vmtDataset/data_qwen_packed.py
qwenvl/data/rope2d.py → vmtDataset/rope2d.py

scripts/sft_7b.sh   →   utils/sft_qwen25VL_7b.sh  
scripts/zero2.json  →   utils/zero2.json
scripts/zero3.json  →   utils/zero3.json



