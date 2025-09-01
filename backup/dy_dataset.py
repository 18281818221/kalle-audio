import os
import json
import torch
import random
import librosa
import numpy as np
import torch.nn as nn

from flows import BigVGANFlowVAE as Generator

from torch.utils.data import Dataset
# from librosa.filters import mel as librosa_mel_fn
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

mel_basis_cache = {}
hann_window_cache = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)

# def mel_spectrogram(
#     y: torch.Tensor,
#     n_fft: int = 1024,
#     num_mels: int = 100,
#     sampling_rate: int = 16000,
#     hop_size: int = 240,
#     win_size: int = 960,
#     fmin: int = 0,
#     fmax: int = None,
#     center: bool = False,
# ) -> torch.Tensor:
#     """
#     Calculate the mel spectrogram of an input signal.
#     This function uses slaney norm for the librosa mel filterbank (using librosa.filters.mel) and uses Hann window for STFT (using torch.stft).

#     Args:
#         y (torch.Tensor): Input signal.
#         n_fft (int): FFT size.
#         num_mels (int): Number of mel bins.
#         sampling_rate (int): Sampling rate of the input signal.
#         hop_size (int): Hop size for STFT.
#         win_size (int): Window size for STFT.
#         fmin (int): Minimum frequency for mel filterbank.
#         fmax (int): Maximum frequency for mel filterbank. If None, defaults to half the sampling rate (fmax = sr / 2.0) inside librosa_mel_fn
#         center (bool): Whether to pad the input to center the frames. Default is False.

#     Returns:
#         torch.Tensor: Mel spectrogram.
#     """
#     if torch.min(y) < -1.0:
#         print(f"[WARNING] Min value of input waveform signal is {torch.min(y)}")
#     if torch.max(y) > 1.0:
#         print(f"[WARNING] Max value of input waveform signal is {torch.max(y)}")

#     device = y.device
#     key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"

#     if key not in mel_basis_cache:
#         mel = librosa_mel_fn(
#             sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
#         )
#         mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
#         hann_window_cache[key] = torch.hann_window(win_size).to(device)

#     mel_basis = mel_basis_cache[key]
#     hann_window = hann_window_cache[key]

#     padding = (n_fft - hop_size) // 2
#     y = torch.nn.functional.pad(
#         y.unsqueeze(1), (padding, padding), mode="reflect"
#     ).squeeze(1)

#     spec = torch.stft(
#         y,
#         n_fft,
#         hop_length=hop_size,
#         win_length=win_size,
#         window=hann_window,
#         center=center,
#         pad_mode="reflect",
#         normalized=False,
#         onesided=True,
#         return_complex=True,
#     )
#     spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

#     mel_spec = torch.matmul(mel_basis, spec)
#     mel_spec = spectral_normalize_torch(mel_spec)

#     return mel_spec
def extract_mel_spec(samples):
    mel_spec, _ = mel_spectogram(
        audio=samples,
        sample_rate=16000,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        n_fft=1024,
        f_min=0,
        f_max=8000,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True
    )

    return mel_spec


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

from aslp.data import FloatNPYData
from aslp.data.mp3data import Mp3Data
from aslp.tools import LanceReader
import io

def get_lance_filelist(input_training_file):
    training_files = []

    # is file
    if os.path.isfile(input_training_file):
        with open(input_training_file, "r", encoding="utf-8") as fi:
            training_lances_paths = [
                x
                # os.path.join(a.input_wavs_dir, x.split("|")[0] + ".wav")
                for x in fi.read().split("\n")
                if len(x) > 0
            ]
    elif os.path.isdir(input_training_file):
        training_lances_paths = [input_training_file]

    datatype2class= {
        "Mp3Data": Mp3Data,
        "FloatNPYData": FloatNPYData,
    }

    for path in training_lances_paths:
        datatype = path.strip().split('_')[-1]
        # assert datatype in datatype2class
        assert datatype == 'Mp3Data',datatype
        reader = LanceReader(path, target_cls=datatype2class[datatype])
        cur_lance_rows = reader.ds.count_rows()
        print(f"lance {path} has {cur_lance_rows} rows")
        for i in range(cur_lance_rows):
            training_files.append((reader, i))

    print(f"total training files: {len(training_files)}")

    return training_files

class TTSDataset_online_lance_for_sft(Dataset):
    def __init__(self, config, tokenizer,base_lst,sft_lst,device):
        
        self.init_vae_generator(config.get('vae_config'))
        self.generator.to(device)
        self.device = device

        self.base_lst = base_lst
        self.sft_lst = sft_lst
        self.set_epoch(0)

        self.length = len(self.info_lines)
        self.pad_token_id = tokenizer.pad_token_id   
        self.tokenizer = tokenizer

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

    def init_vae_generator(self, vae_config):
        config_file = vae_config.get('config_file')
        data = open(config_file).read()
        json_config = json.loads(data)
        self.generator_h = AttrDict(json_config)
        self.generator = Generator(self.generator_h)

        cpt_path = vae_config.get('cpt_path')
        checkpoint_dict = torch.load(cpt_path, map_location='cpu')
        self.generator.load_state_dict(checkpoint_dict['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()

        torch.backends.cudnn.benchmark = False

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.info_lines = self.sft_lst + random.sample(self.base_lst, len(self.sft_lst))
        random.shuffle(self.info_lines)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # info = json.loads(self.info_lines[idx])
        reader, i = self.info_lines[idx]
        datas = reader.get_datas_by_rows([i])
        instance_attributes = datas[0].__dict__.keys()
        if 'data' in instance_attributes:
            utt, audio = datas[0].data_id, datas[0].data
        elif 'mp3_binary' in instance_attributes:
            utt, wav, text = datas[0].data_id, datas[0].mp3_binary , datas[0].text
            wav, _ = librosa.load(io.BytesIO(wav),sr=self.generator_h.sampling_rate,mono=True)
        # utt,text,wav_path = info['utt'],info['text'],info['wav_path']

        text_tokenized = self.tokenizer.encode(text)
        text_ids = torch.from_numpy(np.asarray( text_tokenized + [self.speech_understanding_end_id, self.speech_generation_start_id ] )).long()

        # wav,_ = librosa.load(wav_path, sr=self.generator_h.sampling_rate, mono=True)
        wav = torch.FloatTensor(wav.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            mean_scale_latent = self.generator.extract_latents(wav.unsqueeze(0))

        # mean_scale_latent = torch.from_numpy(np.load(vae_latent_path)).transpose(0,2) # t 128 1 -> 1 128 t
        mean, logs_scale = mean_scale_latent.chunk(2, dim=1)

        stdev = torch.exp(logs_scale)
        latents = torch.randn_like(mean) * stdev + mean

        latents = latents.squeeze(0).transpose(0,1)
        audio_T = latents.size(0)

        total_len = len(text_ids) + audio_T + 1 # 1 is the <|SPEECH_UNDERSTANDING_END|>

        if total_len > 2048:
            print(f"maybe Error in speech_gen_end_idx: {utt}")
            assert 0

        return {
            "input_ids": text_ids,
            "ids_len": text_ids.shape[0],
            "audio_latents": latents,
            "audio_len": latents.shape[0],
            "audio_distribution": mean_scale_latent.squeeze(0).transpose(0,1),
        }

    
    def collate(self, batch):
        b = len(batch)
        distribute_dim = batch[0]['audio_distribution'].shape[1]
        audio_dim = batch[0]['audio_latents'].shape[1]

        max_length = max([i['input_ids'].shape[0]+i['audio_latents'].shape[0] for i in batch])

        input_text_ids = torch.full((b,max_length), self.pad_token_id, dtype=torch.long)
        input_audio_latents = torch.zeros(b,max_length,audio_dim,dtype=torch.float)
        distribute_lables = torch.ones(b,max_length,distribute_dim,dtype=torch.float)

        text_ids_mask = torch.zeros(b,max_length,dtype=torch.bool)
        audio_latents_mask = torch.zeros(b,max_length,dtype=torch.bool)
        distribute_lables_mask = torch.zeros(b,max_length,dtype=torch.bool)
        enddist_mask = torch.zeros(b,max_length,dtype=torch.bool)

        # audio_start_end = []


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
        }

class TTSDataset_online_lance_for_cfg(Dataset):
    def __init__(self, config, tokenizer,base_lst,sft_lst,device):
        
        self.init_vae_generator(config.get('vae_config'))
        self.cfg_prob = config.get('cfg_prob')
        self.generator.to(device)
        self.device = device

        self.base_lst = base_lst
        self.sft_lst = sft_lst
        self.set_epoch(0)

        self.length = len(self.info_lines)
        self.pad_token_id = tokenizer.pad_token_id   
        self.tokenizer = tokenizer

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

    def init_vae_generator(self, vae_config):
        config_file = vae_config.get('config_file')
        data = open(config_file).read()
        json_config = json.loads(data)
        self.generator_h = AttrDict(json_config)
        self.generator = Generator(self.generator_h)

        cpt_path = vae_config.get('cpt_path')
        checkpoint_dict = torch.load(cpt_path, map_location='cpu')
        self.generator.load_state_dict(checkpoint_dict['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()

        torch.backends.cudnn.benchmark = False

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.info_lines = self.sft_lst + random.sample(self.base_lst, len(self.sft_lst))
        random.shuffle(self.info_lines)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # info = json.loads(self.info_lines[idx])
        reader, i = self.info_lines[idx]
        datas = reader.get_datas_by_rows([i])
        instance_attributes = datas[0].__dict__.keys()
        if 'data' in instance_attributes:
            utt, audio = datas[0].data_id, datas[0].data
        elif 'mp3_binary' in instance_attributes:
            utt, wav, text = datas[0].data_id, datas[0].mp3_binary , datas[0].text
            wav, _ = librosa.load(io.BytesIO(wav),sr=self.generator_h.sampling_rate,mono=True)
        # utt,text,wav_path = info['utt'],info['text'],info['wav_path']

        text_tokenized = self.tokenizer.encode(text)
        text_ids = torch.from_numpy(np.asarray( text_tokenized + [self.speech_understanding_end_id, self.speech_generation_start_id ] )).long()

        # wav,_ = librosa.load(wav_path, sr=self.generator_h.sampling_rate, mono=True)
        wav = torch.FloatTensor(wav.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            mean_scale_latent = self.generator.extract_latents(wav.unsqueeze(0))

        # mean_scale_latent = torch.from_numpy(np.load(vae_latent_path)).transpose(0,2) # t 128 1 -> 1 128 t
        mean, logs_scale = mean_scale_latent.chunk(2, dim=1)

        stdev = torch.exp(logs_scale)
        latents = torch.randn_like(mean) * stdev + mean

        latents = latents.squeeze(0).transpose(0,1)
        audio_T = latents.size(0)

        total_len = len(text_ids) + audio_T + 1 # 1 is the <|SPEECH_UNDERSTANDING_END|>

        if total_len > 2048:
            print(f"maybe Error in speech_gen_end_idx: {utt}")
            assert 0

        return {
            "input_ids": text_ids,
            "ids_len": text_ids.shape[0],
            "audio_latents": latents,
            "audio_len": latents.shape[0],
            "audio_distribution": mean_scale_latent.squeeze(0).transpose(0,1),
        }

    
    def collate(self, batch):
        b = len(batch)
        distribute_dim = batch[0]['audio_distribution'].shape[1]
        audio_dim = batch[0]['audio_latents'].shape[1]

        max_length = max([i['input_ids'].shape[0]+i['audio_latents'].shape[0] for i in batch])

        input_text_ids = torch.full((b,max_length), self.pad_token_id, dtype=torch.long)
        input_audio_latents = torch.zeros(b,max_length,audio_dim,dtype=torch.float)
        distribute_lables = torch.ones(b,max_length,distribute_dim,dtype=torch.float)

        text_ids_mask = torch.zeros(b,max_length,dtype=torch.bool)
        audio_latents_mask = torch.zeros(b,max_length,dtype=torch.bool)
        distribute_lables_mask = torch.zeros(b,max_length,dtype=torch.bool)
        enddist_mask = torch.zeros(b,max_length,dtype=torch.bool)

        # audio_start_end = []


        for i,item in enumerate(batch):
            s = item['ids_len']
            e = item['ids_len']+item['audio_len']

            input_text_ids[i,:s] = item['input_ids']
            input_audio_latents[i,s:e] = item['audio_latents']
            distribute_lables[i,s-1:e-1] = item['audio_distribution']

            text_ids_mask[i,:s] = True
            # CFG V1
            # if random.random() < self.cfg_prob: 
            #     text_ids_mask[i,:s] = False

            audio_latents_mask[i,s:e] = True
            for j in range(s,e):
                if random.random() < self.cfg_prob:
                    audio_latents_mask[i,j] = False

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
        }


class TTSDataset_online_lance(Dataset):
    def __init__(self, 
                 config,
                 tokenizer,
                 info_lines,
                 device,
                 output_bf16 = False):
        
        self.init_vae_generator(config.get('vae_config'))
        self.generator.to(device)
        self.device = device

        self.info_lines = info_lines
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

    def init_vae_generator(self, vae_config):
        config_file = vae_config.get('config_file')
        data = open(config_file).read()
        json_config = json.loads(data)
        self.generator_h = AttrDict(json_config)
        self.generator = Generator(self.generator_h)

        cpt_path = vae_config.get('cpt_path')
        checkpoint_dict = torch.load(cpt_path, map_location='cpu')
        self.generator.load_state_dict(checkpoint_dict['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()

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
        # self.info_lines = self.sft_lst + random.sample(self.base_lst, len(self.sft_lst))
        random.seed(epoch)
        random.shuffle(self.info_lines)

    def __getitem__(self, idx):
        # info = json.loads(self.info_lines[idx])
        reader, i = self.info_lines[idx]
        datas = reader.get_datas_by_rows([i])
        instance_attributes = datas[0].__dict__.keys()
        if 'data' in instance_attributes:
            utt, audio = datas[0].data_id, datas[0].data
        elif 'mp3_binary' in instance_attributes:
            utt, wav, text = datas[0].data_id, datas[0].mp3_binary , datas[0].text
            wav, _ = librosa.load(io.BytesIO(wav),sr=self.generator_h.sampling_rate,mono=True)
        # utt,text,wav_path = info['utt'],info['text'],info['wav_path']

        text_tokenized = self.tokenizer.encode(text)
        text_ids = torch.from_numpy(np.asarray( text_tokenized + [self.speech_understanding_end_id, self.speech_generation_start_id ] )).long()

        # wav,_ = librosa.load(wav_path, sr=self.generator_h.sampling_rate, mono=True)
        norm_wav = librosa.util.normalize(wav) * 0.95
        # norm_wav = torch.FloatTensor(norm_wav.reshape(-1)).to(self.device)
        norm_wav = torch.from_numpy(norm_wav)
        mel = extract_mel_spec(norm_wav).unsqueeze(0)
        mel = self.modify_vector(mel).to(self.device)
        
        wav = torch.FloatTensor(wav.reshape(1, -1)).to(self.device)
        
        # mel = mel_spectrogram(norm_wav)
        

        with torch.no_grad():
            mean_scale_latent = self.generator.extract_latents(wav.unsqueeze(0))

        # mean_scale_latent = torch.from_numpy(np.load(vae_latent_path)).transpose(0,2) # t 128 1 -> 1 128 t
        mean, logs_scale = mean_scale_latent.chunk(2, dim=1)

        stdev = torch.exp(logs_scale)
        latents = torch.randn_like(mean) * stdev + mean

        latents = latents.squeeze(0).transpose(0,1)
        audio_T = latents.size(0)

        total_len = len(text_ids) + audio_T + 1 # 1 is the <|SPEECH_UNDERSTANDING_END|>

        # if total_len > 2048:
        #     print(f"maybe Error in speech_gen_end_idx: {utt}")
        #     assert 0

        if self.output_bf16:
            latents = latents.to(torch.bfloat16)
            mean_scale_latent = mean_scale_latent.to(torch.bfloat16)

        return {
            "input_ids": text_ids,
            "ids_len": text_ids.shape[0],
            "audio_latents": latents,
            "audio_len": latents.shape[0],
            "audio_distribution": mean_scale_latent.squeeze(0).transpose(0,1),
            "mel": mel,
        }

    
    def collate(self, batch):
        b = len(batch)
        distribute_dim = batch[0]['audio_distribution'].shape[1]
        audio_dim = batch[0]['audio_latents'].shape[1]
        mel_len = batch[0]['mel'].shape[-1]
        mel_dim = batch[0]['mel'].shape[1]
        latent_dtype = batch[0]['audio_latents'].dtype

        max_length = max([i['input_ids'].shape[0]+i['audio_latents'].shape[0] for i in batch])

        input_text_ids = torch.full((b,max_length), self.pad_token_id, dtype=torch.long)
        input_audio_latents = torch.zeros(b,max_length,audio_dim,dtype=latent_dtype)
        distribute_lables = torch.ones(b,max_length,distribute_dim,dtype=latent_dtype)

        mels = torch.zeros(b,mel_dim,mel_len,dtype=latent_dtype)

        text_ids_mask = torch.zeros(b,max_length,dtype=torch.bool)
        audio_latents_mask = torch.zeros(b,max_length,dtype=torch.bool)
        distribute_lables_mask = torch.zeros(b,max_length,dtype=torch.bool)
        enddist_mask = torch.zeros(b,max_length,dtype=torch.bool)
        speaker_cond_keep = torch.ones(b,dtype=torch.bool)
        for i in range(b):
            if random.random() < self.spk_drop_prob :
                speaker_cond_keep[i] = False

        # audio_start_end = []


        for i,item in enumerate(batch):
            s = item['ids_len']
            e = item['ids_len']+item['audio_len']

            input_text_ids[i,:s] = item['input_ids']
            input_audio_latents[i,s:e] = item['audio_latents']
            distribute_lables[i,s-1:e-1] = item['audio_distribution']

            mels[i] = item['mel'][0]

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
            "mels": mels,
            "speaker_cond_keep": speaker_cond_keep
        }

class Stream_TTSDataset_online_lance(Dataset):
    def __init__(self, 
                 config,
                 tokenizer,
                 info_lines,
                 device,
                 output_bf16 = False):
        
        self.init_vae_generator(config.get('vae_config'))
        self.generator.to(device)
        self.device = device

        self.info_lines = info_lines
        self.set_epoch(0)
        self.length = len(self.info_lines)
        self.pad_token_id = tokenizer.pad_token_id  
        self.delay_frames = config.get('delay_frames') 
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

    def init_vae_generator(self, vae_config):
        config_file = vae_config.get('config_file')
        self.vae_hz = vae_config.get('hz')
        data = open(config_file).read()
        json_config = json.loads(data)
        self.generator_h = AttrDict(json_config)
        self.generator = Generator(self.generator_h)

        cpt_path = vae_config.get('cpt_path')
        checkpoint_dict = torch.load(cpt_path, map_location='cpu')
        self.generator.load_state_dict(checkpoint_dict['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()

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
        # self.info_lines = self.sft_lst + random.sample(self.base_lst, len(self.sft_lst))
        random.seed(epoch)
        random.shuffle(self.info_lines)

    def __getitem__(self, idx):
        # info = json.loads(self.info_lines[idx])
        reader, i = self.info_lines[idx]
        datas = reader.get_datas_by_rows([i])
        instance_attributes = datas[0].__dict__.keys()
        if 'data' in instance_attributes:
            utt, audio = datas[0].data_id, datas[0].data
        elif 'mp3_binary' in instance_attributes:
            utt, wav, text = datas[0].data_id, datas[0].mp3_binary , datas[0].text
            wav, _ = librosa.load(io.BytesIO(wav),sr=self.generator_h.sampling_rate,mono=True)

        text_tokenized = self.tokenizer.encode(text)
        # text_ids = torch.from_numpy(np.asarray( text_tokenized + [self.speech_understanding_end_id, self.speech_generation_start_id ] )).long()
        text_ids = torch.from_numpy(np.asarray( text_tokenized )).long()

        norm_wav = librosa.util.normalize(wav) * 0.95
        norm_wav = torch.from_numpy(norm_wav)
        mel = extract_mel_spec(norm_wav).unsqueeze(0)
        mel = self.modify_vector(mel)

        wav_frames = len(wav) // (self.generator_h.sampling_rate // self.vae_hz)
        
        ids_len = text_ids.shape[0]
        audio_len = wav_frames
        
        # assert audio_len >= ids_len, (audio_len,ids_len,wav.shape)

        if audio_len + self.delay_frames < ids_len:
            print(audio_len,ids_len,wav.shape,text,utt)
            return None

        item_len = audio_len

        return {
            "input_ids": text_ids,
            "ids_len": text_ids.shape[0],
            "mel": mel,
            "wav_samples": wav,
            "item_len": item_len
        }

    
    def collate(self, batch):
        b = len(batch)

        max_wav_sample = max(i['wav_samples'].shape[0] for i in batch)
        delay_sample_num = int( self.delay_frames * (self.generator_h.sampling_rate // self.vae_hz) )
        final_wav_sample = int( max_wav_sample + delay_sample_num  )# delay

        # wav_samples = torch.zeros(b,int(final_wav_sample),dtype=torch.float32)
        wav_samples = np.zeros((b,int(final_wav_sample)),dtype=np.float32)

        for i in range(b):
            wav_samples[i,  delay_sample_num:   delay_sample_num + batch[i]['wav_samples'].shape[0]] = batch[i]['wav_samples']

        wav = torch.FloatTensor(wav_samples).to(self.device)
        with torch.no_grad():
            mean_scale_latent = self.generator.extract_latents(wav.unsqueeze(1))

        mean, logs_scale = mean_scale_latent.chunk(2, dim=1)
        stdev = torch.exp(logs_scale)
        latents = torch.randn_like(mean) * stdev + mean # b d t


        mel_len = batch[0]['mel'].shape[-1]
        mel_dim = batch[0]['mel'].shape[1]
        latent_dtype = batch[0]['mel'].dtype

        # max_length = max(i['item_len'] for i in batch)
        assert latents.shape[2] > max([i['ids_len'] for i in batch])
        max_length = latents.shape[2] - 1

        input_text_ids      = torch.full(   (b,max_length), self.pad_token_id,dtype=torch.long)
        # input_audio_latents = torch.zeros(  b,max_length,audio_dim,           dtype=latent_dtype)
        # distribute_lables   = torch.ones(   b,max_length,distribute_dim,      dtype=latent_dtype)

        mels = torch.zeros(b,mel_dim,mel_len,   dtype=latent_dtype)

        speaker_cond_keep = torch.ones(b,dtype=torch.bool)
        for i in range(b):
            if random.random() < self.spk_drop_prob :
                speaker_cond_keep[i] = False

        # audio_start_end = []


        for i,item in enumerate(batch):
            txt_len   = item['ids_len']

            input_text_ids[i,:txt_len] = item['input_ids']
            mels[i] = item['mel'][0]

        input_audio_latents = latents[:,:,:-1].transpose(1,2)       # -> b t d
        distribute_lables   = mean_scale_latent[:,:,1:].transpose(1,2) # -> b t d
            # text_ids_mask[i,:s] = True
            # audio_latents_mask[i,s:e] = True
            # distribute_lables_mask[i,s-1:e-1] = True
            # enddist_mask[i,e-1:e] = True
            # audio_start_end.append((s,e))
        
        if self.output_bf16:
            input_audio_latents = input_audio_latents.to(torch.bfloat16)
            distribute_lables = distribute_lables.to(torch.bfloat16)
            mels = mels.to(torch.bfloat16)

        return {
            "input_ids": input_text_ids,
            "audio_latents": input_audio_latents,
            "distribute_lables": distribute_lables,

            # "text_ids_mask": text_ids_mask,
            # "audio_latents_mask": audio_latents_mask,
            # "distribute_lables_mask": distribute_lables_mask,
            # "enddist_mask": enddist_mask,
            "mels": mels,
            "speaker_cond_keep": speaker_cond_keep
        }


class TTSDataset_online(Dataset):
    def __init__(self, config, tokenizer,info_lines,device):
        
        self.init_vae_generator(config.get('vae_config'))
        self.generator.to(device)
        self.device = device

        self.info_lines = info_lines
        self.length = len(self.info_lines)
        self.pad_token_id = tokenizer.pad_token_id   
        self.tokenizer = tokenizer
   
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

    def init_vae_generator(self, vae_config):
        config_file = vae_config.get('config_file')
        data = open(config_file).read()
        json_config = json.loads(data)
        self.generator_h = AttrDict(json_config)
        self.generator = Generator(self.generator_h)

        cpt_path = vae_config.get('cpt_path')
        checkpoint_dict = torch.load(cpt_path, map_location='cpu')
        self.generator.load_state_dict(checkpoint_dict['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()

        torch.backends.cudnn.benchmark = False


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        info = json.loads(self.info_lines[idx])
        utt,text,wav_path = info['utt'],info['text'],info['wav_path']

        text_tokenized = self.tokenizer.encode(text)
        text_ids = torch.from_numpy(np.asarray( text_tokenized + [self.speech_understanding_end_id, self.speech_generation_start_id ] )).long()

        wav,_ = librosa.load(wav_path, sr=self.generator_h.sampling_rate, mono=True)
        wav = torch.FloatTensor(wav.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            mean_scale_latent = self.generator.extract_latents(wav.unsqueeze(0))

        # mean_scale_latent = torch.from_numpy(np.load(vae_latent_path)).transpose(0,2) # t 128 1 -> 1 128 t
        mean, logs_scale = mean_scale_latent.chunk(2, dim=1)

        stdev = torch.exp(logs_scale)
        latents = torch.randn_like(mean) * stdev + mean

        latents = latents.squeeze(0).transpose(0,1)
        audio_T = latents.size(0)

        total_len = len(text_ids) + audio_T + 1 # 1 is the <|SPEECH_UNDERSTANDING_END|>

        if total_len > 2048:
            print(f"maybe Error in speech_gen_end_idx: {utt}")
            assert 0

        return {
            "input_ids": text_ids,
            "ids_len": text_ids.shape[0],
            "audio_latents": latents,
            "audio_len": latents.shape[0],
            "audio_distribution": mean_scale_latent.squeeze(0).transpose(0,1),
        }

    
    def collate(self, batch):
        b = len(batch)
        distribute_dim = batch[0]['audio_distribution'].shape[1]
        audio_dim = batch[0]['audio_latents'].shape[1]

        max_length = max([i['input_ids'].shape[0]+i['audio_latents'].shape[0] for i in batch])

        input_text_ids = torch.full((b,max_length), self.pad_token_id, dtype=torch.long)
        input_audio_latents = torch.zeros(b,max_length,audio_dim,dtype=torch.float)
        distribute_lables = torch.ones(b,max_length,distribute_dim,dtype=torch.float)

        text_ids_mask = torch.zeros(b,max_length,dtype=torch.bool)
        audio_latents_mask = torch.zeros(b,max_length,dtype=torch.bool)
        distribute_lables_mask = torch.zeros(b,max_length,dtype=torch.bool)
        enddist_mask = torch.zeros(b,max_length,dtype=torch.bool)

        # audio_start_end = []


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
        }

class TTSDataset(Dataset):
    def __init__(self, config, tokenizer,info_lines):
 
        # memmap_path = os.path.join(data_path, f'{split}_input_ids.memmap')
        # shape_path = os.path.join(data_path, f'{split}_input_ids_shape.npy')

        # self.input_ids = np.memmap(memmap_path, dtype='int32', mode='r', shape=tuple(np.load(shape_path)))
        self.info_lines = info_lines
        self.length = len(self.info_lines)
        self.pad_token_id = tokenizer.pad_token_id   
        self.tokenizer = tokenizer

   
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

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        info = json.loads(self.info_lines[idx])
        utt,text,vae_latent_path = info['utt'],info['text'],info['latent_path']

        text_tokenized = self.tokenizer.encode(text)
        text_ids = torch.from_numpy(np.asarray( text_tokenized + [self.speech_understanding_end_id, self.speech_generation_start_id ] )).long()

        mean_scale_latent = torch.from_numpy(np.load(vae_latent_path)).transpose(0,2) # t 128 1 -> 1 128 t
        mean, logs_scale = mean_scale_latent.chunk(2, dim=1)

        stdev = torch.exp(logs_scale)
        latents = torch.randn_like(mean) * stdev + mean

        latents = latents.squeeze(0).transpose(0,1)
        audio_T = latents.size(0)

        total_len = len(text_ids) + audio_T + 1 # 1 is the <|SPEECH_UNDERSTANDING_END|>

        if total_len > 2048:
            print(f"maybe Error in speech_gen_end_idx: {utt}")
            assert 0

        return {
            "input_ids": text_ids,
            "ids_len": text_ids.shape[0],
            "audio_latents": latents,
            "audio_len": latents.shape[0],
            "audio_distribution": mean_scale_latent.squeeze(0).transpose(0,1),
        }

    
    def collate(self, batch):
        b = len(batch)
        distribute_dim = batch[0]['audio_distribution'].shape[1]
        audio_dim = batch[0]['audio_latents'].shape[1]

        max_length = max([i['input_ids'].shape[0]+i['audio_latents'].shape[0] for i in batch])

        input_text_ids = torch.full((b,max_length), self.pad_token_id, dtype=torch.long)
        input_audio_latents = torch.zeros(b,max_length,audio_dim,dtype=torch.float)
        distribute_lables = torch.ones(b,max_length,distribute_dim,dtype=torch.float)

        text_ids_mask = torch.zeros(b,max_length,dtype=torch.bool)
        audio_latents_mask = torch.zeros(b,max_length,dtype=torch.bool)
        distribute_lables_mask = torch.zeros(b,max_length,dtype=torch.bool)
        enddist_mask = torch.zeros(b,max_length,dtype=torch.bool)

        # audio_start_end = []


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
        }


if __name__ == "__main__":
    import yaml
    from transformers import AutoTokenizer
    from tqdm import tqdm
    
    config_file = "./configs/test.yaml"
    device = "cuda"
    config = yaml.safe_load(open(config_file))
    base_lst = get_lance_filelist(config["dataset"]["meta_path"])

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    train_dataset = Stream_TTSDataset_online_lance(config["dataset"],tokenizer,base_lst,device,output_bf16=config['use_flash_attation'])

    for l in tqdm(range(2000000)):
        reader,i = train_dataset.info_lines[l]
        datas = reader.get_datas_by_rows([i])
        utt,wav,text = datas[0].data_id,datas[0].mp3_binary,datas[0].text
        wav,_ = librosa.load(io.BytesIO(wav),sr=train_dataset.generator_h.sampling_rate,mono=True)

        text_tokenized = train_dataset.tokenizer.encode(text)
        text_ids = torch.from_numpy(np.asarray( text_tokenized )).long()
        wav_frames = len(wav) // (train_dataset.generator_h.sampling_rate // train_dataset.vae_hz)

        ids_len = text_ids.shape[0]
        audio_len = wav_frames

        if audio_len < ids_len:
            print('???')
            import pdb;pdb.set_trace()


        

    batch = [train_dataset[0],train_dataset[1],train_dataset[2]]
    batch = train_dataset.collate(batch)

    import pdb;pdb.set_trace()

    print(len(train_dataset))