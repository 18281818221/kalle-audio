
export HF_HOME="/mnt/bn/twj-data-multimodal2/workspace/hf_cache"
export TRANSFORMERS_CACHE="/mnt/bn/twj-data-multimodal2/workspace/hf_cache"
export HUGGINGFACE_HUB_CACHE="/mnt/bn/twj-data-multimodal2/workspace/hf_cache"

export NCCL_DEBUG=INFO 
export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# export CUDA_VISIBLE_DEVICES="0,1,2,3"


# 读取 CUDA_VISIBLE_DEVICES 环境变量
VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-""}
# 计算 GPU 数量（通过逗号分割后的元素个数）
if [ -z "$VISIBLE_DEVICES" ]; then
    # 如果未设置，默认使用所有可用 GPU（通过 nvidia-smi 获取）
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
else
    # 按逗号分割并统计数量
    GPU_COUNT=$(echo "$VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi
# export NCCL_TIMEOUT=180s  # NCCL 通信超时（默认 30s）
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 打印更详细的分布式日志，便于排查
# /mnt/bn/twj-data-multimodal2/environment/anaconda2/bin/python
accelerate launch --config_file ./default_config.yaml \
                 --main_process_port 12338 \
                 --num_processes ${GPU_COUNT} train_offline.py
