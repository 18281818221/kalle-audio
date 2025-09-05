import os, sys, json, shutil, time, random, yaml, io, librosa
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset
# from librosa.filters import mel as librosa_mel_fn
from torch import nn
from twj_utils import read_parquet, vae_sample, read_jsonl
from twj_utils import AttrDict


class TTSDataset_online_parquet(Dataset):
    def __init__(self, 
                 config,
                 tokenizer,
                 info_lines,
                 device,
                 generator,
                 output_bf16 = False):
        

        self.device = device

        
        print('start reading parquet')
        
        # self.info_lines = read_parquet(info_lines)

        print('start reading jsonl')

        self.info_lines = read_jsonl(info_lines)
        random.shuffle(self.info_lines)
        print(f'total datalines: {len(self.info_lines)}')
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

        self.target_sr = 16000

        print('dataset init end')

        self.init_vae_generator(config.get('vae_config'))
        # self.generator = generator
    def init_vae_generator(self, vae_config):
        vae_model_config = vae_config.get('config_file')
        vae_cpkt_path = vae_config.get('cpt_path')

        print('vae_model_config,', vae_model_config)
        print('vae_cpkt_path', vae_cpkt_path)

        with open(vae_model_config) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        import sys
        sys.path.append('/mnt/bn/twj-data-multimodal2/workspace/melvae')
        from models_flow_vae import BigVGANFlowVAE as Generator
        self.generator = Generator(h)
        def load_checkpoint(filepath, device):
            assert os.path.isfile(filepath)
            print("Loading '{}'".format(filepath))
            checkpoint_dict = torch.load(filepath, map_location=device)
            print("Complete.")
            return checkpoint_dict
        state_dict_g = load_checkpoint(vae_cpkt_path, "cpu")
        miss, unexpected = self.generator.load_state_dict(state_dict_g['generator'], strict=False)
        print("Missing keys:", miss)
        print("Unexpected keys:", unexpected)
        for param in self.generator.parameters():
            param.requires_grad = False
        self.generator.eval()
        print("self.generator load complete.")

        torch.backends.cudnn.benchmark = False

    def __len__(self):
        return self.length
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        random.seed(epoch)

    def __getitem__(self, idx):

        while True:

            try:
                item = self.info_lines[idx]
                # 如果item为空
                if not item or (isinstance(item, str) and item.strip() == ""):
                    idx = random.randint(0, len(self.info_lines)-1)
                    continue
                speech_path = item['speech']
                # video_path = item['video']
                if "AudioSetCaps" not in item.keys():
                    text = item['caption']
                else:
                    text = item['AudioSetCaps']
                data_id = item['id']
                # vae_latent_path = item['vae_latent_path']
                # vae_latent_path = item['vae']    # latents ([1, 1024, 105])


# VAE latent online
                vae_latent_path = os.path.splitext(speech_path)[0] + '.melvae.npy'
                if os.path.exists(vae_latent_path):
                    mean_scale_latent = np.load(vae_latent_path)
                    mean_scale_latent = torch.from_numpy(mean_scale_latent)
                else:
                    wav, sr = librosa.load(speech_path, sr=self.target_sr, mono=True)
                    wav = torch.from_numpy(wav).reshape(1, -1).unsqueeze(0) # 1,1,t
                    with torch.no_grad():
                        mean_scale_latent = self.generator.extract_latents(wav) # mean_scale_latent torch.Size([1, 1024, 105])
                mean, logs_scale = mean_scale_latent.chunk(2, dim=1)
                stdev = torch.exp(logs_scale)
                latents = torch.randn_like(mean) * stdev + mean  # latents torch.Size([1, 512, 125])
                latents = latents.transpose(1,2) # 1, dim, T -> # 1, T, Dim
                mean_scale_latent = mean_scale_latent.transpose(1,2) # 1, dim, T -> # 1, T, Dim

# VAE latent offline
                # mean_scale_latent, latents = self.get_stableaudio_latent(vae_latent_path)
                # mean_scale_latent, latents = self.get_sigmaVAE_latent(vae_latent_path) # (1, 76, 64), (1, 76, 64),
                # mean_scale_latent, latents = self.get_melVAE_latent(vae_latent_path) # latents ([1, 1024, 105]) 


        # text handing:

                text_tokenized = self.tokenizer.encode(text)
                text_ids = torch.from_numpy(np.asarray( text_tokenized + [self.speech_understanding_end_id, self.speech_generation_start_id ] )).long()



                if self.output_bf16:
                    latents = latents.to(torch.bfloat16)
                    mean_scale_latent = mean_scale_latent.to(torch.bfloat16)


                # total_len = len(text_ids) + audio_T + 1 # 1 is the <|SPEECH_UNDERSTANDING_END|>

                return_dict = {
                    "input_ids": text_ids,
                    "ids_len": text_ids.shape[0],
                    "audio_latents": latents,
                    "audio_len": latents.shape[1],
                    "audio_distribution": mean_scale_latent,
                    "mel": None,
                    "raw_text": text,
                    "speech_path": speech_path,
                }
                for key, val in return_dict.items():
                    if isinstance(val, torch.Tensor) and (torch.isnan(val).any() or torch.isinf(val).any()):
                        raise (f"输入 {data_id} 包含无效值")
            except Exception as e:
                idx = random.randint(0, len(self.info_lines)-1)
                print (f'get item error: {e}. new idx item:{idx}')
                continue

            return return_dict

    
    def collate(self, batch):
        b = len(batch)
        distribute_dim = batch[0]['audio_distribution'].shape[2]
        audio_dim = batch[0]['audio_latents'].shape[2]

        latent_dtype = batch[0]['audio_latents'].dtype
        max_length = max([i['input_ids'].shape[0]+i['audio_latents'].shape[1] for i in batch])

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


        raw_texts = []
        speech_paths = []
        
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

            raw_texts.append(item['raw_text'])
            speech_paths.append(item['speech_path'])


        return_dict = {
            "input_ids": input_text_ids,
            "audio_latents": input_audio_latents,
            "distribute_lables": distribute_lables,
            "text_ids_mask": text_ids_mask,
            "audio_latents_mask": audio_latents_mask,
            "distribute_lables_mask": distribute_lables_mask,
            "enddist_mask": enddist_mask,
            "mels": None,
            "speaker_cond_keep": speaker_cond_keep,
            "raw_texts": raw_texts,
            "speech_paths": speech_paths
        }

        return return_dict


    def get_stableaudio_latent(vae_latent_path):
        mean_scale_latent = np.load(vae_latent_path) # 128, t
        if len(mean_scale_latent.shape)==2:
            mean_scale_latent = mean_scale_latent.unsqueeze(0)
        # mean_scale_latent = torch.randn(1, 512, 200)
        mean_scale_latent = torch.from_numpy(mean_scale_latent) # 1, 128, T
        mean, scale = mean_scale_latent.chunk(2, dim=1)
        latents, kl = vae_sample(mean, scale) 
        if len(latents.shape)==3:
            latents = latents.squeeze(0)      # 1, dim, T -> # dim, T 
        latents = latents.transpose(0,1)    # dim, T -> T, dim
        return mean_scale_latent, latents



    def get_melVAE_latent(self, vae_latent_path):
        mean_scale_latent = np.load(vae_latent_path) # latents ([1, 1024, 105])
        mean_scale_latent = torch.from_numpy(mean_scale_latent) # latents ([1, 1024, 105])

        mean, logs_scale = mean_scale_latent.chunk(2, dim=1)
        stdev = torch.exp(logs_scale)
        latents = torch.randn_like(mean) * stdev + mean  # # latents ([1, 1024, 105])

        return mean_scale_latent, latents


    def get_sigmaVAE_latent(self, vae_latent_path):
        mean_scale_latent = np.load(vae_latent_path) # 'shape': (1, 76, 64),
        mean_scale_latent = torch.from_numpy(mean_scale_latent) # (1, 76, 64),

        latents = mean_scale_latent

        return mean_scale_latent, latents
    