import argparse
import os
import json
from sacrebleu.metrics import BLEU
import jieba
import torch
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
# from comet import download_model, load_from_checkpoint
import evaluate
import numpy as np
import warnings
import sys

# 抑制 multiprocess 在 Python 3.12 中的资源追踪器警告
warnings.filterwarnings("ignore", category=ResourceWarning)
if sys.version_info >= (3, 12):
    import os
    os.environ['PYTHONWARNINGS'] = 'ignore::ResourceWarning' 

def computeBLEU(preds, refs, isZh=False, usingSacreBLEU=True):
    if usingSacreBLEU:
        # 使用sacreBLEU包
        refs = [refs]
        bleuMetric = BLEU()
        if isZh:
            bleuMetric = BLEU(tokenize='zh')
        return bleuMetric.corpus_score(preds, refs).score
    else:
        # 使用huggingface的evaluate中的sacreBLEU
        refs = [[ref] for ref in refs]
        bleuMetric = evaluate.load("sacrebleu")
        if isZh:
            return bleuMetric.compute(predictions=preds, references=refs, tokenize='zh')["score"]
        else:
            return bleuMetric.compute(predictions=preds, references=refs)["score"]

def computeMETEOR(preds, refs, isZh=False):
    if isZh:
        chinese_tokenizer = lambda text: list(jieba.cut(text))
        tokenized_preds = [chinese_tokenizer(pred) for pred in preds]
        tokenized_refs = [chinese_tokenizer(ref) for ref in refs]
        scores = [meteor_score([ref], pred) for pred, ref in zip(tokenized_preds, tokenized_refs)]
        return sum(scores) / len(scores) if scores else 0
    else:
        meteor = evaluate.load('meteor')
        return meteor.compute(predictions=preds, references=refs)['meteor']

def computeChrF(preds, refs):
    refs = [[ref] for ref in refs]
    chrfMetric = evaluate.load("chrf")
    return chrfMetric.compute(predictions=preds, references=refs)['score']

def computeCOMET(src, preds, refs):

    def compute(src, preds, refs):
        torch.set_float32_matmul_precision("high")
        comet_metric = evaluate.load('comet')
        comet_score = comet_metric.compute(predictions=preds, references=refs, sources=src)
        return comet_score

    def setNetwork(proxyAddress):
        
        os.environ["http_proxy"] = proxyAddress
        os.environ["https_proxy"] = proxyAddress
        os.environ["all_proxy"] = proxyAddress
        
    try:
        return compute(src, preds, refs)
    except:
        print("change network_address")
        proxy_addresses = ["http://172.18.31.59:7890", "http://10.5.29.44:7897"]
        for address in proxy_addresses:
            try:
                setNetwork(address)
                return compute(src, preds, refs)
            except Exception as e:
                print(f"address {address} network error")
                print(f"{str(e)}")
        print("All network address failed.COMET is not computed.")

def computeBLEURT(preds, refs, batchSize=512, returnAverage=True):
    BLEURTModel = BleurtForSequenceClassification.from_pretrained('./huggingface/lucadiliello/BLEURT-20')
    BLEURTTokenizer = BleurtTokenizer.from_pretrained('./huggingface/lucadiliello/BLEURT-20')
    BLEURTModel = BLEURTModel.to('cuda')
    BLEURTModel.eval()
    
    bleurt_scores = []  # 存储每个样本的BLEURT值
    with torch.no_grad():
        for i in range(0, len(refs), batchSize):
            batch_refs = refs[i:i + batchSize]
            batch_preds = preds[i:i + batchSize]
            inputs = BLEURTTokenizer(batch_refs, batch_preds, padding='longest', truncation=True, max_length=512, return_tensors='pt')
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            logits = BLEURTModel(**inputs).logits.flatten()  # 每个样本的打分
            bleurt_scores.extend(logits.cpu().numpy().tolist())  # 将每个样本的打分添加到列表中
    if returnAverage:
        return sum(bleurt_scores) / len(bleurt_scores) if bleurt_scores else 0.0
    else:
        return bleurt_scores

def detect_language_is_Chinese(text):
    # 判断每个字符的Unicode值来确定语言
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # 中文字符的Unicode范围
            return True
    return False

def cleanLLMLongTranslate(predText):
    """清理提取的翻译文本"""
    text = text.strip()
    
    # 1. 检测重复模式
    if len(text) > 200:
        first_part = text[:50]
        first_occurrence = text.find(first_part)
        second_occurrence = text.find(first_part, first_occurrence + 1)
        
        if second_occurrence != -1 and second_occurrence < len(text) // 2:
            text = text[:second_occurrence].strip()
    
    # 2. 移除HTML标签残留
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&lt;[^&]*&gt;', '', text)
    
    # 3. 清理多余的换行和空格
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # 4. 如果文本仍然异常长，尝试找到合理的截断点
    if len(text) > 300:
        for i, char in enumerate(text):
            if char in '。！？.!?' and i > 50:
                if i < 200:
                    text = text[:i+1]
                    break
        else:
            text = text[:200]
    
    return text.strip()

def getSrcPredsRefs(DirName):
    with open(os.path.join(DirName, "results.json"), 'r') as f:
        prdsData = json.load(f)
    src, preds, refs = [], [], []
    for data in prdsData:
        src.append(data['src'])
        preds.append(data['preds'])
        refs.append(data['refs'])
    
    if "</think>" in preds[0] and "</think>" in preds[1]:
        preds = [cleanLLMLongTranslate(pred.split("</think>")[-1].strip()) for pred in preds]
    elif "<translation>" in preds[0] and "<translation>" in preds[1]:
        preds = [cleanLLMLongTranslate(pred.split("<translation>")[-1].strip()) for pred in preds]

    return src, preds, refs

def computeTranslationMetrics(DirName, save_comet_scores=False, metrics = ['BLEU', 'METEOR', 'chrF', 'COMET', 'BLEURT']):
    src, preds, refs = getSrcPredsRefs(DirName)
    
    with open(os.path.join(DirName, "usingForTransMetric.json"), 'w') as f:
        json.dump([{"src": itemSrc, "preds": itemPreds, "refs": itemRefs} for itemSrc, itemPreds, itemRefs \
            in zip(src, preds, refs)], f, ensure_ascii=False, indent=4)
    
    tgtIsChinese = detect_language_is_Chinese(refs[0])

    metricScores = []

    if 'BLEU' in metrics:
        BLEUScore = computeBLEU(preds, refs, tgtIsChinese, usingSacreBLEU=True)
        print( f"\033[91m BLEU: {BLEUScore}    ( Using SacreBLEU )   \033[0m" )
        metricScores.append(BLEUScore)
    if 'METEOR' in metrics:
        METEORScore = computeMETEOR(preds, refs, tgtIsChinese) * 100
        print( f"\033[91m METEOR: {METEORScore} \033[0m" )
        metricScores.append(METEORScore)
    if 'chrF' in metrics:
        chrFScore = computeChrF(preds, refs)
        print( f"\033[91m chrF: {chrFScore} \033[0m" )
        metricScores.append(chrFScore)
    if 'COMET' in metrics:
        COMETScore = computeCOMET(src, preds, refs)
        
        MeanCOMETScore = COMETScore["mean_score"] * 100
        print( f"\033[91m COMET: {MeanCOMETScore} \033[0m" )
        metricScores.append(MeanCOMETScore)
        if save_comet_scores:
            np.save(os.path.join(DirName, "comet_scores.npy"), COMETScore["scores"])
        
    if 'BLEURT' in metrics:
        BLEURTScore = computeBLEURT(preds, refs) * 100
        print( f"\033[91m BLEURT: {BLEURTScore} \033[0m" )
        metricScores.append(BLEURTScore)

    assert len(metricScores) == len(metrics), "The length of metric scores and metrics should be equal."
    with open(os.path.join(DirName, "translationMetricScores.log"), 'w') as f:
        f.write(f"Evaluation Resutlts \n")
        for metric, score in zip(metrics, metricScores):
            f.write(f"{metric}: {score}\n")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir_path", type=str, required=True)
    parser.add_argument("-m", "--metrics", nargs='+', default=['BLEU', 'METEOR', 'chrF', 'COMET', 'BLEURT'],
                        help="Specify which metrics to compute. Available options: BLEU, METEOR, chrF, COMET, BLEURT. "
                                "Default is to compute all metrics.")
    parser.add_argument("-sc", "--save_comet_scores", action="store_true")
    args = parser.parse_args()
