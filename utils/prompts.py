import json

# 增加了多模态模型的systemprompt
def getSystemPrompt(modelName, modelType,promptType):
    if promptType == "translation":
        if modelType == "text":
            return  {"role": "system", "content": "You are an expert translator who is fluent in English and Chinese."}
        elif modelType == "multimodal":
            return {"role": "system","content": [{"type": "text", "text": "You are an expert translator who is fluent in English and Chinese."},],}
        else:
            raise TypeError("Model type format error!")
    modelName2Type = {"Qwen2-7B-Instruct": "qwen2", "Qwen2.5-7B-Instruct": "qwen2", "Qwen2.5-14B-Instruct": "qwen2", "Llama-3-8B-Instruct": "llama3", \
        "LLaVA-NeXT-Video-7B-hf":"llava", "Qwen2-VL-7B-Instruct":"qwen2-vl","MiniCPM-V-2_6":"minicpm","Qwen2.5-VL-7B-Instruct":"qwen2.5-vl","Qwen2.5-VL-32B-Instruct":"qwen2.5-vl",\
            "Qwen2.5-VL-3B-Instruct":"qwen2.5-vl", "internlm3-8b-instruct":"internlm3", "Qwen2.5-3B-Instruct": "qwen2", "Qwen3-4B": "qwen2", "Qwen3-8B": "qwen2", "Qwen3-30B-A3B":"qwen2",\
        "InternVideo2_5_Chat_8B":"InternVideo2_5","Llama-3.2-11B-Vision-Instruct":"llama3.2-vision","Llama-3.1-8B-Instruct":"llama3.1", "Qwen3-32B": "qwen2"}
    systemPrompts = dict()
    systemPrompts["qwen2"] = {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
    systemPrompts["llama3"] = None    
    systemPrompts["llava"] = None  
    systemPrompts["qwen2-vl"] = None
    systemPrompts["minicpm"] = None
    systemPrompts["qwen2.5-vl"] = None
    systemPrompts["internlm3"] = {"role": "system", "content": """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."""}
    systemPrompts["InternVideo2_5"] = None
    systemPrompts["llama3.2-vision"] = None
    systemPrompts["llama3.1"] = None
    return systemPrompts[modelName2Type[modelName]]

def getUserPrompt(promptLanguage, srcLanguage, tgtLanguage, srcSent, shotNum=0, dataset_type="text", prompt_type=None):
    userPrompts = dict()
    sentencePairsOfshot = [
        {'zh': '请问最近的地铁站在哪里？', 'en': 'Excuse me, where is the nearest subway station?'},\
        {'zh': '很高兴见到你。', 'en': "It's nice to meet you."},\
        {'zh': '祝你有美好的一天！', 'en': "I wish you a wonderful day!"},\
        {'zh': '不要放弃，坚持就是胜利。', 'en': "Don't give up; persistence is victory."},\
        {'zh': '你让我很开心。', 'en': "You make me very happy."},\
        {'zh': '我想点一份牛排。', 'en': "I would like to order a steak."},\
        {'zh': '今天的会议几点开始？', 'en': "What time does today’s meeting start?"},\
        {'zh': '当然可以，我很乐意帮助你。', 'en': "Of course, I'd be happy to help you."},\
    ]
    languageID2text = {"zh": {"zh": "中文", "en": "英文"}, "en": {"zh": "Chinese", "en": "English"}}
    if dataset_type == "text":
        userPrompts["zh"] = f"请把以下输入的句子从{languageID2text['zh'][srcLanguage]}翻译成{languageID2text['zh'][tgtLanguage]}。请只输出翻译后的句子。\n"
        userPrompts["en"] = f"Please translate the following input sentence from {languageID2text['en'][srcLanguage]} to {languageID2text['en'][tgtLanguage]}. ONLY output the translated sentence.\n"
        for i in range(shotNum):
            userPrompts["zh"] += f"输入句子:\n{sentencePairsOfshot[i][srcLanguage]}\n翻译：\n{sentencePairsOfshot[i][tgtLanguage]}\n"
            userPrompts["en"] += f"Input sentence:\n{sentencePairsOfshot[i][srcLanguage]}\nTranslation sentence:\n{sentencePairsOfshot[i][tgtLanguage]}\n"
        userPrompts["zh"] += f"输入句子：\n{srcSent}\n翻译后的句子：\n"
        userPrompts["en"] += f"Input sentence:\n{srcSent}\nTranslated sentence:\n"
        if prompt_type == "promptYang":
            # 如果是yang的prompt，重新编写
            userPrompts["en"] = f"Please translate the following sentences into {languageID2text['en'][tgtLanguage]}. The input sentences are wrapped by <sentence> and </sentence>:\n"
            userPrompts["en"] += f"\n<sentence>\n{srcSent}\n</sentence>\n"
            userPrompts["en"] += "\nThe translated result should be wrapped by <translated> and </translated>."
    elif dataset_type == "video-text":
        if shotNum != 0:
            raise TypeError("Only zero shot is supported now in video-text")
        userPrompts["zh"] = f"请根据输入的视频内容，把以下输入的句子从{languageID2text['zh'][srcLanguage]}翻译成{languageID2text['zh'][tgtLanguage]}。请只输出翻译后的句子。\n"
        userPrompts["en"] = f"Please translate the following input sentence from {languageID2text['en'][srcLanguage]} to {languageID2text['en'][tgtLanguage]} according to the video. ONLY output the translated sentence.\n"
        userPrompts["zh"] += f"输入句子：\n{srcSent}\n翻译后的句子：\n"
        userPrompts["en"] += f"Input sentence:\n{srcSent}\nTranslated sentence:\n"
        if prompt_type == "newVideoTextPromptTest":
            userPrompts["en"] = f"You are provided with a video context. Your task is to translate the given input sentence from {languageID2text['zh'][srcLanguage]} into {languageID2text['zh'][tgtLanguage]}, strictly based on the video's content.\n"
            userPrompts["en"] += f"Requirements:\nONLY output the translated sentence.\nEnsure the translation is consistent with the video's context and meaning.\n\n" 
            userPrompts["en"] += f"Input sentence:\n{srcSent}\nTranslated sentence:\n"
    elif dataset_type == 'image-text':
        if shotNum != 0:
            raise TypeError("Only zero shot is supported now in video-text")
        if prompt_type is not None:
            if prompt_type == "usingInferenceImage":
                userPrompts["zh"] = "我将给你一个输入句子，它是一个视频片段的字幕，我还会同时输入这个视频片段中和这个句子最相关的画面。"
                userPrompts["en"] = "I will give you an input sentence, which is a subtitle of a video clip, and I will also input the frame that is most relevant to this sentence in this video clip. "
                userPrompts["zh"] = f"请根据输入的图片内容，把输入的句子从{languageID2text['zh'][srcLanguage]}翻译成{languageID2text['zh'][tgtLanguage]}。请只输出翻译后的句子。\n"
                userPrompts["en"] = f"Please translate the input sentence from {languageID2text['en'][srcLanguage]} to {languageID2text['en'][tgtLanguage]} based on the input image content. Please ONLY output the translated sentence.\n"
                userPrompts["zh"] += f"输入句子：\n{srcSent}\n翻译后的句子：\n"
                userPrompts["en"] += f"Input sentence:\n{srcSent}\nTranslated sentence:\n"
        else:
            userPrompts["zh"] = f"请根据输入的图片内容，把以下输入的句子从{languageID2text['zh'][srcLanguage]}翻译成{languageID2text['zh'][tgtLanguage]}。请只输出翻译后的句子。\n"
            userPrompts["en"] = f"Please translate the following input sentence from {languageID2text['en'][srcLanguage]} to {languageID2text['en'][tgtLanguage]} according to the image. ONLY output the translated sentence.\n"
            userPrompts["zh"] += f"输入句子：\n{srcSent}\n翻译后的句子：\n"
            userPrompts["en"] += f"Input sentence:\n{srcSent}\nTranslated sentence:\n"
    elif dataset_type == 'images-text':   # 给模型多张图片，让模型自己选择图片进行推理
        if shotNum != 0:
            raise TypeError("Only zero shot is supported now in images-text")
        if prompt_type is not None:
            if prompt_type == "chooseImage":
                userPrompts["zh"] = "我将给你一个输入句子，它是一个视频片段的字幕，我还会同时输入这个视频片段的十帧画面。"
                userPrompts["en"] = "I will give you an input sentence, which is a subtitle of a video clip, and I will also input the frames of this video clip."
                userPrompts["zh"] += f"我需要将这个输入句子从{languageID2text['zh'][srcLanguage]}翻译成{languageID2text['zh'][tgtLanguage]}，请你从这十帧画面中选取和这个句子最相关的一帧画面，也即对翻译输入句子最有用的一帧画面。\n"
                userPrompts["en"] += f"I need to translate this input sentence from {languageID2text['zh'][srcLanguage]} to {languageID2text['zh'][tgtLanguage]}. Please select the frame that is most relevant to this sentence from these ten frames, that is, the frame that is most useful for translating the input sentence.\n"
                userPrompts["zh"] += "请只输出画面的编号，如第4张画面和翻译句子最相关，则输出4。\n"
                userPrompts["en"] += "Please ONLY output the frame number, such as the fourth frame is most relevant to the translated sentence, then output 4.\n"
                userPrompts["zh"] += f"输入句子：\n{srcSent}\n"
                userPrompts["en"] += f"Input sentence:\n{srcSent}\n"
                userPrompts["zh"] += "选取的画面序号：\n"
                userPrompts["en"] += "Selected frame number:\n"
            else:
                raise TypeError("Prompt type error!")
        else:
            userPrompts["zh"] = f"我将给你一个输入句子，它是一个视频片段的字幕，我还会同时输入这个视频片段的十帧画面。"
            userPrompts["en"] = f"I will give you an input sentence, which is a subtitle of a video clip, and I will also input the uniformly sampled frames of this video clip. "
            userPrompts["zh"] += f"请从输入的十帧画面中选取对翻译有用的画面信息，以将输入的句子从{languageID2text['zh'][srcLanguage]}翻译成{languageID2text['zh'][tgtLanguage]}。请只输出翻译后的句子。\n"
            userPrompts["en"] += f"Please select the useful image information for translation from the input frames to translate the input sentence from {languageID2text['en'][srcLanguage]} to {languageID2text['en'][tgtLanguage]}. ONLY output the translated sentence.\n"
            userPrompts["zh"] += f"输入句子：\n{srcSent}\n翻译后的句子：\n"
            userPrompts["en"] += f"Input sentence:\n{srcSent}\nTranslated sentence:\n"
    else:
        raise TypeError("Dataset type error!")
    return userPrompts[promptLanguage]

if __name__ == '__main__':
    print(getUserPrompt('en', 'zh', 'en', "我爱中国", 3))