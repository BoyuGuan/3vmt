


def get_prompts(prmptType, language=None):
    allPropmts = dict()
    allPropmts["videoCaption"] = {
        "zh": "向我描述一下这个视频",
        "en": "Describe this video to me"
    }


    return allPropmts[prmptType][language]










