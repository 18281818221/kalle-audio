
# export CUDA_VISIBLE_DEVICES="4"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export HF_HOME="/mnt/bn/twj-data-multimodal2/workspace/hf_cache"
export NCCL_DEBUG=INFO 
export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1

# 启动训练时设置环境变量
export NCCL_TIMEOUT=180s  # NCCL 通信超时（默认 30s）
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 打印更详细的分布式日志，便于排查
# /home/node57_data2/kxxia/env/anaco
# /home/node57_data2/kxxia/env/anaconda3/envs/py39/bin/python
accelerate launch --config_file ./default_config_cpu.yaml \
                 --main_process_port 12366 \
                 --num_processes 1 train.py
