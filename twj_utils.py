import pyarrow.parquet as pq
import io
import pandas as pd
from torch import nn
import torch
import json


def read_jsonl( jsonl_list ):
    data = []
    for jsonl in jsonl_list:
        print(f'read jsonl {jsonl}')
        with open(jsonl, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    return data

def vae_sample(mean, scale):
    stdev = nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean
    kl = (mean * mean + var - logvar - 1).sum(1).mean()
    return latents, kl

def get_mean_stdev_from_stableaudio2_latents(mean_scale_latent):
    # mean_scale_latent torch.Size([2, 261, 128])
    mean, scale = mean_scale_latent.chunk(2, dim=1)
    stdev = nn.functional.softplus(scale) + 1e-4
    return mean, stdev

def read_parquet(parquet_paths):

    # 定义多个路径（支持文件夹、文件列表或通配符）
    # parquet_paths = [
    #     "/mnt/bn/twj-data-multimodal2/libritts_r/data/test.clean/",
    #     "/mnt/bn/twj-data-multimodal2/libritts_r/data/train.other.500",
    #     "/mnt/bn/twj-data-multimodal2/libritts_r/data/train.clean.360",
    #     "/mnt/bn/twj-data-multimodal2/libritts_r/data/train.clean.100"
    #     # "/path/to/folder_with_parquets/"  # 文件夹下的所有 Parquet 文件
    # ]
    total_df = pd.DataFrame()
    for dir_name in parquet_paths:
        dataset = pq.ParquetDataset(dir_name)   
        table = dataset.read()
        df = table.to_pandas()
        total_df = pd.concat([total_df, df], axis=0)

    return total_df
        # audio_bytes = df.iloc[0]['audio']['bytes']  # 提取二进制数据
        # text_normalized = df.iloc[0]['text_normalized']
        # data_id = df.iloc[0]['id']
        # # 将二进制数据转换为音频流（类似文件对象）
        # audio_io = io.BytesIO(audio_bytes)

        # # 方法2：用librosa读取（更适合后续音频特征处理）
        # # 重置流指针到开头
        # audio_io.seek(0)  # 重置流指针到开头
        # waveform, sample_rate = librosa.load(audio_io, sr=None)  # sr=None保留原始采样率
        # print(waveform.shape)
        # print(sample_rate)
