import os
import json
import torch
import random
import librosa
import numpy as np
import torch.nn as nn
import io
from torch.utils.data import Dataset
# from librosa.filters import mel as librosa_mel_fn
from torch import nn
from twj_utils import read_parquet, vae_sample
mel_basis_cache = {}
hann_window_cache = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)

def extract_mel_spec(device):
    # mel_spec, _ = mel_spectogram(
    #     audio=samples,
    #     sample_rate=16000,
    #     hop_length=256,
    #     win_length=1024,
    #     n_mels=80,
    #     n_fft=1024,
    #     f_min=0,
    #     f_max=8000,
    #     power=1,
    #     normalized=False,
    #     min_max_energy_norm=True,
    #     norm="slaney",
    #     mel_scale="slaney",
    #     compression=True
    # )
    # 换成torch提取mel
    import torchaudio
    # 参数与原来保持一致
    sample_rate = 16000
    hop_length = 256
    win_length = 1024
    n_mels = 80
    n_fft = 1024
    f_min = 0
    f_max = 8000
    power = 1
    compression = True

    # 使用 transform（每次创建开销较小，但可按需缓存）
    melspec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
        f_min=f_min,
        f_max=f_max,
        norm='slaney',
        mel_scale='slaney'
    )
    # .to(device)

    return melspec_transform


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DynamicBatchGenerator():
    def __init__(self,hparams,epoch=0,disable_bar=True):
        # self.dataset = dataset
        self.epoch = epoch
        self.hparams = hparams

        self.batch_size = self.hparams.get("batch_size",9999999)
        self.use_dynamic = self.hparams["use_dynamic"]
        self.max_token_length = self.hparams["max_token_length"]

        self.cur_batch_max_len = 0
        self.cur_batch = [] 

        self.count = 0

    def batch_add_item(self,item):
        if item is None:
            return None
        if self.use_dynamic:
            item_len = item["item_len"] # 获得当前item的长度
            tmp_len = max(item_len,self.cur_batch_max_len) # 输入总token是 最大的 * batch_size

            if tmp_len* (len(self.cur_batch)+1) <= self.max_token_length and len(self.cur_batch) < self.batch_size: # 还可以填充
                self.cur_batch.append(item)
                self.cur_batch_max_len = tmp_len

            else:   # 再加入就满了
                output_batch = self.cur_batch
                self.cur_batch = [item] if item_len < self.max_token_length else []
                self.cur_batch_max_len = item_len if item_len < self.max_token_length else 0
                return output_batch

        else:
            self.cur_batch.append(item)
            if len(self.cur_batch) >= self.batch_size:
                output_batch = self.cur_batch
                self.cur_batch = []
                return output_batch
            
        return None

def dict_to_device(feature_dict,device):
    result = {}
    for key in feature_dict:
        if isinstance(feature_dict[key],torch.Tensor):
            result[key] = feature_dict[key].to(device)
        else:
            result[key] = feature_dict[key]
    return result

def get_vae_h(config_path):
    data = open(config_path).read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    return h

class TTSDataset_online_parquet(Dataset):
    def __init__(self, 
                 config,
                 tokenizer,
                 info_lines,
                 device,
                 output_bf16 = False):
        

        self.device = device

        
        print('start reading parquet')
        
        self.info_lines = read_parquet(info_lines)

        print('reading end')

        self.set_epoch(0)
        self.length = len(self.info_lines)
        self.pad_token_id = tokenizer.pad_token_id   
        self.tokenizer = tokenizer
        self.output_bf16 = output_bf16

        self.speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')    # 128260
        self.speech_generation_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')        # 128261
        self.text_generation_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_START|>')        # 128256
        self.text_generation_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_END|>')            # 128257
        self.text_understanding_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_START|>')  # 128258
        self.text_understanding_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_END|>')      # 128259
        self.speech_understanding_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')  # 128262
        self.speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')      # 128263
 
        self.max_length = 2048
        self.ignore_index = -100  
        self.spk_drop_prob = config.get('spk_drop_prob', 0.0)

        self.mel_transformer = extract_mel_spec(device)
        self.target_sr = 44100

        print('dataset init end')

        self.init_vae_generator(config.get('vae_config'))
        # self.generator.to(device)
    def init_vae_generator(self, vae_config):
        vae_model_config = vae_config.get('config_file')
        vae_cpkt_path = vae_config.get('cpt_path')

        print('vae_model_config,', vae_model_config)
        print('vae_cpkt_path', vae_cpkt_path)

        with open(vae_model_config) as f:
            vae_model_config = json.load(f)
        from stable_audio_tools.models import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict

        self.generator = create_model_from_config(vae_model_config)

        print(' start loading ckpt from ', vae_cpkt_path)
        ckpt = load_ckpt_state_dict(vae_cpkt_path)
        
        miss, unexpected = self.generator.load_state_dict(ckpt, strict=False)
        print("Missing keys:", miss)
        print("Unexpected keys:", unexpected)
        for param in self.generator.parameters():
            param.requires_grad = False
        self.generator.eval()

        torch.backends.cudnn.benchmark = False

    def modify_vector(self, vector_bdt, size=200):
        # 560 ~ 6s
        if vector_bdt.shape[-1] < size:
            while vector_bdt.shape[-1] < size:
                vector_bdt = vector_bdt.repeat(1,1,2)
            vector_bdt = vector_bdt[:,:,:size]
        elif vector_bdt.shape[-1] > size:
            start_index = np.random.randint(low=0, high=vector_bdt.shape[-1] - size)
            vector_bdt = vector_bdt[:,:,start_index:start_index + size]
        return vector_bdt

    def __len__(self):
        return self.length
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        random.seed(epoch)

    def __getitem__(self, idx):
        # info = json.loads(self.info_lines[idx])
        item = self.info_lines.iloc[idx]

        audio_bytes = item['audio']['bytes']  # 提取二进制数据
        text = item['text_normalized']
        data_id = item['id']
        # vae_latent_path = item['vae_latent_path']



# Audio stream: extract VAE latent
        audio_io = io.BytesIO(audio_bytes)
        audio_io.seek(0)  # 重置流指针到开头
        # 修改1: 使用target_sr统一采样率
        wav, sample_rate = librosa.load(audio_io, sr=self.target_sr, mono=True)  # sr=None保留原始采样率
        norm_wav = librosa.util.normalize(wav) * 0.95
        norm_wav = torch.from_numpy(norm_wav).reshape(1, -1)
        dual_norm_wav = norm_wav.repeat(2, 1).unsqueeze(0)
        
        mean_scale_latent = self.generator.pretransform.encode(dual_norm_wav)# 1, 2, T --> 1, 128, T
        mean_scale_latent = mean_scale_latent.detach()

# VAE latent
        # mean_scale_latent = np.load(vae_latent_path) # 128, t

        # print('mean_scale_latent', mean_scale_latent.shape)

        if len(mean_scale_latent.shape)==2:
            mean_scale_latent = mean_scale_latent.unsqueeze(0)

        # mean_scale_latent = torch.randn(1, 512, 200)
        mean, scale = mean_scale_latent.chunk(2, dim=1)
        latents, kl = vae_sample(mean, scale) 
        if len(latents.shape)==3:
            latents = latents.squeeze(0)      # 1, dim, T -> # dim, T 
        latents = latents.transpose(0,1)    # dim, T -> T, dim

        # mean, logs_scale = mean_scale_latent.chunk(2, dim=1)
        # stdev = torch.exp(logs_scale)
        # latents = torch.randn_like(mean) * stdev + mean
        # latents = latents.squeeze(0).transpose(0,1)
        # audio_T = latents.size(0)

        
        # print('latents', latents.shape) # T, dim


# text handing:

        text_tokenized = self.tokenizer.encode(text)
        text_ids = torch.from_numpy(np.asarray( text_tokenized + [self.speech_understanding_end_id, self.speech_generation_start_id ] )).long()



        if self.output_bf16:
            latents = latents.to(torch.bfloat16)
            mean_scale_latent = mean_scale_latent.to(torch.bfloat16)


        # total_len = len(text_ids) + audio_T + 1 # 1 is the <|SPEECH_UNDERSTANDING_END|>

        return {
            "input_ids": text_ids,
            "ids_len": text_ids.shape[0],
            "audio_latents": latents,
            "audio_len": latents.shape[0],
            "audio_distribution": mean_scale_latent.squeeze(0).transpose(0,1),
            "mel": None,
        }

    
    def collate(self, batch):
        b = len(batch)
        distribute_dim = batch[0]['audio_distribution'].shape[1]
        audio_dim = batch[0]['audio_latents'].shape[1]

        latent_dtype = batch[0]['audio_latents'].dtype
        max_length = max([i['input_ids'].shape[0]+i['audio_latents'].shape[0] for i in batch])

        input_text_ids = torch.full((b,max_length), self.pad_token_id, dtype=torch.long)
        input_audio_latents = torch.zeros(b,max_length,audio_dim,dtype=latent_dtype)
        distribute_lables = torch.ones(b,max_length,distribute_dim,dtype=latent_dtype)

        

        text_ids_mask = torch.zeros(b,max_length,dtype=torch.bool)
        audio_latents_mask = torch.zeros(b,max_length,dtype=torch.bool)
        distribute_lables_mask = torch.zeros(b,max_length,dtype=torch.bool)
        enddist_mask = torch.zeros(b, max_length,dtype=torch.bool)
        speaker_cond_keep = torch.ones(b,dtype=torch.bool)
        for i in range(b):
            if random.random() < self.spk_drop_prob :
                speaker_cond_keep[i] = False



        for i,item in enumerate(batch):
            s = item['ids_len']
            e = item['ids_len']+item['audio_len']

            input_text_ids[i,:s] = item['input_ids']
            input_audio_latents[i,s:e] = item['audio_latents']
            distribute_lables[i,s-1:e-1] = item['audio_distribution']

            text_ids_mask[i,:s] = True
            audio_latents_mask[i,s:e] = True
            distribute_lables_mask[i,s-1:e-1] = True
            enddist_mask[i,e-1:e] = True
            # audio_start_end.append((s,e))

        return {
            "input_ids": input_text_ids,
            "audio_latents": input_audio_latents,
            "distribute_lables": distribute_lables,

            "text_ids_mask": text_ids_mask,
            "audio_latents_mask": audio_latents_mask,
            "distribute_lables_mask": distribute_lables_mask,
            "enddist_mask": enddist_mask,
            "mels": None,
            "speaker_cond_keep": speaker_cond_keep
        }


if __name__ == "__main__":
    import yaml
    from transformers import AutoTokenizer
    from tqdm import tqdm
    
    config_file = "/mnt/bn/twj-data-multimodal2/workspace/kalle/configs/twj.yaml"
    device = "cpu"
    config = yaml.safe_load(open(config_file))
        

    print(f'use parquet: {base_lst}')

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])

    print('create dataset ')
    train_dataset = TTSDataset_online_parquet(config["dataset"],tokenizer,[base_lst],device,output_bf16=config['use_flash_attation'])

    print('dataset size', len(train_dataset))

    for l in tqdm(range(2000000)):
        item = train_dataset.info_lines.iloc[l]
        audio_bytes = item['audio']['bytes']  # 提取二进制数据
        text = item['text_normalized']
        data_id = item['id']
        # 将二进制数据转换为音频流（类似文件对象）
        audio_io = io.BytesIO(audio_bytes)
        audio_io.seek(0)  # 重置流指针到开头
        wav, sample_rate = librosa.load(audio_io, sr=train_dataset.target_sr, mono=True)  # sr=None保留原始采样率
        print('wav', wav.shape)



        text_tokenized = train_dataset.tokenizer.encode(text)
        text_ids = torch.from_numpy(np.asarray( text_tokenized + [train_dataset.speech_understanding_end_id, train_dataset.speech_generation_start_id ] )).long()


        print('text_ids', text_ids)

        
        # wav,_ = librosa.load(wav_path, sr=self.generator_h.sampling_rate, mono=True)
        norm_wav = librosa.util.normalize(wav) * 0.95
        # norm_wav = torch.FloatTensor(norm_wav.reshape(-1)).to(self.device)
        norm_wav = torch.from_numpy(norm_wav)
        mel = train_dataset.mel_transformer(norm_wav).unsqueeze(0)
        print('mel', mel.shape)
        mel = train_dataset.modify_vector(mel).to(train_dataset.device)
        
        wav = torch.FloatTensor(wav.reshape(1, -1)).to(train_dataset.device)
        wav = wav.repeat(2, 1)
        print('wav', wav.shape)
        
        # mel = mel_spectrogram(norm_wav)

        with torch.no_grad():
            mean_scale_latent = train_dataset.generator.pretransform.encode(wav.unsqueeze(0))
        tmp_latent = mean_scale_latent.detach().cpu().numpy()
        np.save( os.path.join("/mnt/bn/twj-data-multimodal2/workspace/kalle/tmp_dir", f"{l}.npy"), tmp_latent)


        # mean_scale_latent = torch.from_numpy(np.load(vae_latent_path)).transpose(0,2) # t 128 1 -> 1 128 t
        mean, logs_scale = mean_scale_latent.chunk(2, dim=1)

        stdev = torch.exp(logs_scale)
        latents = torch.randn_like(mean) * stdev + mean

        print('latents' ,   latents.shape)
        print('mean_scale_latent' ,   mean_scale_latent.shape)

        latents = latents.squeeze(0).transpose(0,1)
        audio_T = latents.size(0)

        total_len = len(text_ids) + audio_T + 1 # 1 is the <|SPEECH_UNDERSTANDING_END|>

        if train_dataset.output_bf16:
            latents = latents.to(torch.bfloat16)
            mean_scale_latent = mean_scale_latent.to(torch.bfloat16)



        

    batch = [train_dataset[0],train_dataset[1],train_dataset[2]]
    batch = train_dataset.collate(batch)

    import pdb;pdb.set_trace()

    print(len(train_dataset))