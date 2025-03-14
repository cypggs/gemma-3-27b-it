export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
pip install -U vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U "vllm[cu121]" -i https://pypi.tuna.tsinghua.edu.cn/simple

huggingface-cli download google/gemma-3-27b-it \
  --local-dir /root/autodl-tmp/gemma-3-27b-it \
  --local-dir-use-symlinks False \
  --resume-download
  
# 1. 创建并写入文件（使用 Here Document 语法）
cat > gemma_vllm_api.py << 'EOF'
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from fastapi import FastAPI
from pydantic import BaseModel

# 配置引擎参数
engine_args = AsyncEngineArgs(
    model="./gemma-3-27b-it",
    tokenizer="./gemma-3-27b-it",
    tensor_parallel_size=2,  # 根据 GPU 数量调整
    dtype="bfloat16",        # 3090/A10 建议用 float16
    trust_remote_code=True,
    gpu_memory_utilization=0.9
)

# 初始化引擎
engine = AsyncLLMEngine.from_engine_args(engine_args)

app = FastAPI()

class Query(BaseModel):
    prompt: str
    max_tokens: int = 512

@app.post("/generate")
async def generate(query: Query):
    results = await engine.generate(
        prompt=query.prompt,
        sampling_params={
            "max_tokens": query.max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
    )
    return {"response": results[0].outputs[0].text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6006)
EOF

# 2. 验证文件内容（使用带行号查看）
cat -n gemma_vllm_api.py

# 3. 启动服务（需先完成模型下载）
CUDA_VISIBLE_DEVICES=0 python gemma_vllm_api.py &
  
curl -X POST "http://localhost:6006/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "如何做红烧肉？", "max_tokens": 300}'
