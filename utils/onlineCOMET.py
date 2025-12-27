import os
import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ---------------------------------------------------------
# 0. 安装pip依赖
# pip install fastapi uvicorn
# ---------------------------------------------------------

# ---------------------------------------------------------
# 1. 配置命令行参数 (在 import torch/comet 之前处理最好)
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="COMET Metric Server")
parser.add_argument("--device", type=str, default="3", help="指定使用的显卡ID，例如 '0' 或 '1'")
parser.add_argument("--port", type=int, default=10086, help="服务监听端口")
args = parser.parse_args()

# 设置环境变量，必须在模型加载前设置
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
print(f"=== 配置: 使用显卡 CUDA_VISIBLE_DEVICES={args.device} (服务端口: {args.port}) ===")

# ---------------------------------------------------------
# 2. 导入模型库 (设置完环境变量后再导入)
# ---------------------------------------------------------
from comet import load_from_checkpoint

# 定义请求数据结构
class CometRequest(BaseModel):
    src: List[str]
    preds: List[str]
    refs: List[str]

app = FastAPI()

# ---------------------------------------------------------
# 3. 加载模型 (常驻显存)
# ---------------------------------------------------------
# 请确保路径正确
MODEL_PATH = "./huggingface/Unbabel/wmt22-comet-da/checkpoints/model.ckpt"

print(f"正在加载模型到 GPU (映射后ID: 0)...")
try:
    # 加载模型
    comet_model = load_from_checkpoint(MODEL_PATH)
    # 强制将模型移动到 CUDA (由于设置了VISIBLE_DEVICES，这里用 cuda:0 即可指向指定卡)
    comet_model = comet_model.cuda()
    comet_model.eval() # 设置为评估模式
    print("模型加载完成！")
except Exception as e:
    print(f"CRITICAL ERROR: 模型加载失败: {e}")
    exit(1)

@app.post("/compute_score")
async def compute_score(request: CometRequest):
    if not (len(request.src) == len(request.preds) == len(request.refs)):
        raise HTTPException(status_code=400, detail="输入列表长度不一致")
    
    data = [
        {"src": src_i, "mt": preds_i, "ref": refs_i} 
        for src_i, preds_i, refs_i in zip(request.src, request.preds, request.refs)
    ]
    
    try:
        # predict 函数通常接受 gpus 参数。
        # 因为我们要么已经用 .cuda() 移动了模型，要么通过环境变量限制了可见卡，
        # 这里的 gpus=1 表示使用"当前可见的1张卡"。
        # 注意：某些版本的 comet 可能不需要显式 gpus=1，如果报错可改为 gpus=0 或去掉。
        model_output = comet_model.predict(data, batch_size=8, gpus=1)
        
        return {
            "system_score": model_output.system_score,
            "scores": model_output.scores
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理出错: {str(e)}")

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=args.port)