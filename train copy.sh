
export CUDA_VISIBLE_DEVICES="0"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export HF_HOME="/mnt/bn/twj-data-multimodal2/workspace/hf_cache"
export TRANSFORMERS_CACHE="/mnt/bn/twj-data-multimodal2/workspace/hf_cache"
export HUGGINGFACE_HUB_CACHE="/mnt/bn/twj-data-multimodal2/workspace/hf_cache"

export NCCL_DEBUG=INFO 
export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1

# export NCCL_TIMEOUT=180s  # NCCL 通信超时（默认 30s）
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 打印更详细的分布式日志，便于排查
# /mnt/bn/twj-data-multimodal2/environment/anaconda2/bin/python
accelerate launch --config_file ./default_config.yaml \
                 --main_process_port 12368 \
                 --num_processes 1 train_offline.py
