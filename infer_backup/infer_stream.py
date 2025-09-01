
import os
import yaml
import json
import torch
import librosa
import argparse
import numpy as np
import torch.distributions as D
import random
from torch import nn
from transformers import AutoModelForCausalLM

from tqdm import tqdm
from ecapa_tdnn import ECAPA_TDNN
# from model import Llasa
from flows import BigVGANFlowVAE as Generator
from transformers import AutoTokenizer,get_scheduler
from scipy.io.wavfile import write

from dy_dataset import extract_mel_spec

def modify_vector(vector_bdt, size=200):
    # 560 ~ 6s
    if vector_bdt.shape[-1] < size:
        while vector_bdt.shape[-1] < size:
            vector_bdt = vector_bdt.repeat(1,1,2)
        vector_bdt = vector_bdt[:,:,:size]
    elif vector_bdt.shape[-1] > size:
        start_index = np.random.randint(low=0, high=vector_bdt.shape[-1] - size)
        vector_bdt = vector_bdt[:,:,start_index:start_index + size]
    return vector_bdt

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Llasa_text_stream_spkvae(nn.Module):
    def __init__(
        self,
        config,
        tokenizer,
        use_flash_attention = False
    ):
        super().__init__()

        self.use_fa = use_flash_attention
        if self.use_fa:
            self.base_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                config['llm_model_name_or_path'],
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
        else:
            self.base_model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                config['llm_model_name_or_path'],
                torch_dtype=torch.float32,
            )
        self.base_model.resize_token_embeddings(len(tokenizer))
        self.base_model.vocab_size = len(tokenizer)

        self.vocab_size = self.base_model.config.vocab_size
        self.hidden_size = self.base_model.config.hidden_size
        self.audio_linear = nn.Linear(config['latent_dim'],2048,dtype=torch.bfloat16) if self.use_fa else nn.Linear(config['latent_dim'],2048)
        self.distribution_linear = nn.Linear(2048,config['latent_dim']*2,dtype=torch.bfloat16) if self.use_fa else nn.Linear(2048,config['latent_dim']*2)
        # self.mrte = MRTE()
        # if self.use_fa:
        #     self.mrte = self.mrte.to(torch.bfloat16)
        self.speaker_encoder = ECAPA_TDNN(in_channels=80, channels=512, embd_dim=2048)
        self.speaker_cond_disp_linear = nn.Linear(2048,2048*2)
        if self.use_fa:
            self.speaker_encoder = self.speaker_encoder.to(torch.bfloat16)
            self.speaker_cond_disp_linear = self.speaker_cond_disp_linear.to(torch.bfloat16)

    def forward(
        self,
        input_ids,              # b,t
        audio_latents,          # b,t,d1
        audio_distribution_l,   # b,t,d2
        attation_mask,
        target_mask,
        enddist_mask,
        mels,                   # b,d,t
        bos_mask,
        bos_token,

    ):
        # text embedding
        text_embed = self.base_model.model.embed_tokens(input_ids)  # b,t,d
        audio_embed = self.audio_linear(audio_latents)              # b,t,d
        bos_embed = self.base_model.model.embed_tokens(bos_token)
        audio_embed[bos_mask] = bos_embed
        audio_latents_dim = audio_latents.shape[-1]

        speaker_cond = self.speaker_encoder(mels.transpose(1,2))
        speaker_cond_disp = self.speaker_cond_disp_linear(speaker_cond)
        speaker_cond_mean,speaker_cond_logs_scale = speaker_cond_disp.chunk(2,dim=1)
        speaker_cond = torch.randn_like(speaker_cond_mean) * torch.exp(speaker_cond_logs_scale) + speaker_cond_mean

        norm_disp = D.Normal(torch.zeros_like(speaker_cond_mean),torch.ones_like(speaker_cond_logs_scale))
        speaker_disp = D.Normal(speaker_cond_mean,  torch.exp(speaker_cond_logs_scale))
        speaker_cond_kl = D.kl_divergence(speaker_disp ,norm_disp).sum(1) / 2048
        speaker_cond_kl = speaker_cond_kl.sum() / speaker_cond_kl.shape[0]

        input_embed = text_embed + audio_embed
        input_embed = torch.concat((speaker_cond.unsqueeze(1),input_embed),dim=1)

        true_column = torch.ones((attation_mask.shape[0], 1), dtype=attation_mask.dtype, device=attation_mask.device)
        attation_mask = torch.cat((true_column,attation_mask ), dim=1)

        hidden = self.base_model.model(
            inputs_embeds=input_embed,
            attention_mask=attation_mask,
        )[0]  # b,t,d
        hidden = hidden[:,1:,:]

        distribution_p = self.distribution_linear(hidden)       # b,t,d2

        mean1,logs_scale1 = audio_distribution_l.chunk(2,dim=2)
        mean2,logs_scale2 = distribution_p.chunk(2,dim=2)

        std1 = torch.exp(logs_scale1)
        std2 = torch.exp(logs_scale2)

        l_disp = D.Normal(mean1,std1) # 均值和标准差
        p_disp = D.Normal(mean2,std2)

        kl = D.kl_divergence(p_disp, l_disp)

        kl = kl.sum(2) / audio_latents_dim
        # audio_loss = kl.mean()
        audio_loss = (kl * target_mask).sum() / target_mask.sum()
        end_loss = (kl * enddist_mask).sum() / enddist_mask.sum()

        return {
            "speaker_cond_kl": speaker_cond_kl,
            "audio_loss": audio_loss,
            "end_loss": end_loss,
            "pre_mean": mean2,
            "pre_log_scale": logs_scale2
        }
    
    
    @torch.no_grad()
    def infer(
        self,
        input_ids,              # t
        audio_latents,          # 
        mels,                   # b,d,t

        bos_token,
        txt_pad_token,
        max_length = 1000,
    ):
        text_embed = self.base_model.model.embed_tokens(input_ids.unsqueeze(0))
        audio_embed = self.audio_linear(audio_latents) 
        bos_embed = self.base_model.model.embed_tokens(bos_token.unsqueeze(0)).unsqueeze(0)
        audio_embed = torch.cat((audio_embed,bos_embed),dim=1)

        if mels is not None:
            speaker_cond = self.speaker_encoder(mels.transpose(1,2))
            speaker_cond_disp = self.speaker_cond_disp_linear(speaker_cond)
            speaker_cond_mean,speaker_cond_logs_scale = speaker_cond_disp.chunk(2,dim=1)
            speaker_cond = torch.randn_like(speaker_cond_mean) * torch.exp(speaker_cond_logs_scale)
        else:
            zero_mean = torch.zeros((1,2048),dtype=text_embed.dtype,device=text_embed.device)
            one_scale = torch.ones((1,2048),dtype=text_embed.dtype,device=text_embed.device)
            speaker_cond = torch.randn_like(zero_mean) * one_scale

        assert text_embed.shape[1] >= audio_embed.shape[1]
        audio_len = audio_embed.shape[1]
        input_embed = text_embed[:,:audio_len,:] + audio_embed

        input_embed = torch.concat((speaker_cond.unsqueeze(1),input_embed),dim=1)
        final_audio_latents_lst = []

        for i in range(audio_len+1,input_ids.shape[0]):
            hidden = self.base_model.model(inputs_embeds=input_embed)
            last_hidden = hidden[0][:,-1:,:]
            last_disp = self.distribution_linear(last_hidden)

            mean,logs_scale2 = last_disp.chunk(2,dim=2)
            gen_audio_latent = torch.randn_like(mean) * torch.exp(logs_scale2) + mean

            gen_audio_embed = self.audio_linear(gen_audio_latent)
            gen_input_embed = text_embed[:,i:i+1,:] + gen_audio_embed
            input_embed = torch.cat((input_embed,gen_input_embed),dim=1)


            # audio_latents = torch.cat((audio_latents,gen_audio_latent),dim=1)

            final_audio_latents_lst.append(last_disp)

            end_disp = D.Normal(torch.ones_like(mean),torch.exp(torch.ones_like(logs_scale2))) # 均值和标准差
            p_disp = D.Normal(mean,torch.exp(logs_scale2))
            latent_dim = mean.shape[2] 
            kl = D.kl_divergence(p_disp, end_disp).sum(2) / latent_dim
            if kl < 0.5 and i > 3:
                break

            # audio_embed = self.audio_linear(audio_latents)
            # input_embed = text_embed[:,:i,:] + audio_embed
           
        generate_audio_latents = torch.stack(final_audio_latents_lst[:-1],dim=1).squeeze(1).squeeze(2)
        return generate_audio_latents.transpose(1,2)



class infer_tools:
    def __init__(self, 
                 config, 
                 device, 
                 test_mod, 
                 check_point_path,
                 use_prompt=False,
                 streaming=False,
                 icl=False,
                 utt2prompt_wav=None,
                 utt2prompt_txt=None,
                 utt2target_txt=None):
        self.init_config(config)
        
        self.test_mod = test_mod
        self.device = f"cuda:{device}"
        self.check_point_path = check_point_path

        self.use_prompt = use_prompt
        self.streaming = streaming
        self.icl = icl

        self.utt2prompt_wav = utt2prompt_wav
        self.utt2prompt_txt = utt2prompt_txt
        self.utt2target_txt = utt2target_txt

        self.check_point_path = check_point_path
        if self.check_point_path is None:
            checkpoints = [os.path.join(self.config['output_dir'], d) for d in os.listdir(self.config['output_dir']) if d.startswith("epoch_")]
            self.check_point_path = max(checkpoints, key=os.path.getmtime)

        self.init_vae_generator()
        self.init_tokenizer()
        self.init_model()
        self.init_output_info()

        self.init_target_info()
        

    def init_config(self, config):
        config = yaml.safe_load(open(config))
        self.config = config
        self.config['exp_dir'] = os.path.join(config['exp_dir'],config['project_name'])
        self.config['output_dir'] = os.path.join(config['exp_dir'],'output')
        self.delay_frames = config['dataset']['delay_frames']
        self.vae_hz = config['dataset']['vae_config']['hz']

    def init_vae_generator(self):
        vae_model_config = self.config['dataset']['vae_config'].get('config_file')
        with open(vae_model_config) as f:
            vae_model_config = f.read()

        json_config = json.loads(vae_model_config)
        self.h = AttrDict(json_config)
        generator = Generator(self.h)

        file_path = self.config['dataset']['vae_config'].get('cpt_path')
        state_dict_g = torch.load(file_path, map_location='cpu')
        generator.load_state_dict(state_dict_g['generator'])
        generator.eval()
        generator.remove_weight_norm()
        generator.to(self.device)
        torch.backends.cudnn.benchmark = False
        self.generator = generator

    def init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer_path'])
        self.speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')    # 128260
        self.speech_generation_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')        # 128261
        self.text_generation_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_START|>')        # 128256
        self.text_generation_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_END|>')            # 128257
        self.text_understanding_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_START|>')  # 128258
        self.text_understanding_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_END|>')      # 128259
        self.speech_understanding_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')  # 128262
        self.speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')      # 128263
        self.tokenizer = tokenizer

    def init_model(self):
        model = Llasa_text_stream_spkvae(self.config['model'],self.tokenizer)
        self.latent_dim = self.config['model']['latent_dim']
        self.ckpt = torch.load(self.check_point_path,map_location='cpu')
        model.load_state_dict(self.ckpt)
        for name, param in model.named_parameters():
            assert param.dtype == torch.float32
        self.model = model.to(self.device)

    def reinit_model(self):
        self.model.load_state_dict(self.ckpt)
        self.model = self.model.to(self.device)


    def init_output_info(self):
        version = os.path.basename(os.path.dirname(os.path.dirname(self.check_point_path)))
        ckpt_name = os.path.basename(self.check_point_path)
        self.target_name = f"test-{version}-{ckpt_name}-{self.test_mod}"
        if self.use_prompt:
            self.target_name += '-prompt'
        if self.icl:
            self.target_name += '-icl'
        if self.streaming:
            self.target_name += '-streaming'
        os.makedirs(self.target_name, exist_ok=True)

    def init_target_info(self):
        if self.utt2target_txt is not None:
            if self.use_prompt:
                assert len([utt for utt in self.utt2target_txt if utt not in self.utt2prompt_txt]) == 0
                assert len([utt for utt in self.utt2target_txt if utt not in self.utt2prompt_wav]) == 0
            return
        self.utt2target_txt = {}
        self.utt2prompt_txt = {}
        self.utt2prompt_wav = {}
        if self.test_mod == 'hard':
            test_dir = os.path.join('./test_seed_dir', 'zh')
            test_meta = os.path.join(test_dir, 'hardcase.lst')
        else:
            test_dir = os.path.join('./test_seed_dir', self.test_mod)
            test_meta = os.path.join(test_dir, 'meta.lst')
        
        meta_lst = open(test_meta).readlines()
        for meta in meta_lst:
            meta = meta.strip()
            utt,prompt_text,prompt_wav,target_text = meta.split('|')
            self.utt2prompt_txt[utt] = prompt_text
            self.utt2prompt_wav[utt] = os.path.join(test_dir, prompt_wav)
            self.utt2target_txt[utt] = target_text

    def infer(self):
        self.model.eval()
        for utt in tqdm(self.utt2target_txt):
            if os.path.exists(os.path.join(self.target_name, f"{utt}.wav")):
                continue

            target_text = self.utt2target_txt[utt]
            target_text_tokenized = self.tokenizer.encode(target_text)
            prompt_wav = self.utt2prompt_wav[utt]
            if self.use_prompt:

                prompt_wav,_ = librosa.load(prompt_wav, sr=self.h.sampling_rate, mono=True)
                norm_wav = librosa.util.normalize(prompt_wav) * 0.95
                norm_wav = torch.from_numpy(norm_wav)
                mel = extract_mel_spec(norm_wav).unsqueeze(0)
                mel = modify_vector(mel)

                if self.icl:

                    prompt_text = self.utt2prompt_txt[utt]
                    prompt_text_tokenized = self.tokenizer.encode(prompt_text)

                    target_text_tokenized = target_text_tokenized[1:]
                    
                    if self.streaming:
                        delay_frames = min( self.delay_frames, len(prompt_text_tokenized) + len(target_text_tokenized) )
                    else:
                        delay_frames = len(prompt_text_tokenized) + len(target_text_tokenized)

                    delay_sample_num = int( delay_frames * (self.h.sampling_rate // self.vae_hz) )
                    prompt_wav = np.pad(prompt_wav,(delay_sample_num,0),mode='constant',constant_values=0)

                    prompt_wav_tensor = torch.FloatTensor(prompt_wav.reshape(1, -1)).to(self.device)
                    with torch.no_grad():
                        prompt_mean_scale_latent = self.generator.extract_latents(prompt_wav_tensor.unsqueeze(0))
                    mean, logs_scale = prompt_mean_scale_latent[:,:,:-1].transpose(1,2).chunk(2, dim=2)
                    audio_latents = torch.randn_like(mean) * torch.exp(logs_scale) + mean
                    
                else:

                    prompt_text_tokenized = []
                    if self.streaming:
                        delay_frames = min( self.delay_frames, len(target_text_tokenized) )
                    else:
                        delay_frames = len(target_text_tokenized)

                    delay_sample_num = int( delay_frames * (self.h.sampling_rate // self.vae_hz) )
                    prompt_wav = np.zeros((1,int(delay_sample_num)),dtype=np.float32)

                    prompt_wav_tensor = torch.FloatTensor(prompt_wav).to(self.device)
                    with torch.no_grad():
                        prompt_mean_scale_latent = self.generator.extract_latents(prompt_wav_tensor.unsqueeze(0))
                    mean, logs_scale = prompt_mean_scale_latent[:,:,:-1].transpose(1,2).chunk(2, dim=2)
                    audio_latents = torch.randn_like(mean) * torch.exp(logs_scale) + mean

            else:
                mel = None
                prompt_text_tokenized = []

                if self.streaming:
                    delay_frames = min( self.delay_frames, len(target_text_tokenized) )
                else:
                    delay_frames = len(target_text_tokenized)

                delay_sample_num = int( delay_frames * (self.h.sampling_rate // self.vae_hz) )
                prompt_wav = np.zeros((1,int(delay_sample_num)),dtype=np.float32)

                prompt_wav_tensor = torch.FloatTensor(prompt_wav).to(self.device)
                with torch.no_grad():
                    prompt_mean_scale_latent = self.generator.extract_latents(prompt_wav_tensor.unsqueeze(0))
                mean, logs_scale = prompt_mean_scale_latent[:,:,:-1].transpose(1,2).chunk(2, dim=2)
                audio_latents = torch.randn_like(mean) * torch.exp(logs_scale) + mean

            # if self.use_prompt:

            #     target_text_tokenized = self.tokenizer.encode(target_text)

            #     prompt_wav = self.utt2prompt_wav[utt]

            #     # delay_sample_num = int( self.delay_frames * (self.h.sampling_rate // self.vae_hz) )
            #     delay_frames = min( self.delay_frames, len(target_text_tokenized) )
            #     delay_sample_num = int( delay_frames * (self.h.sampling_rate // self.vae_hz) )
            #     import pdb;pdb.set_trace()

            #     wav_samples = np.zeros((1,int(delay_sample_num)),dtype=np.float32)
            #     wav = torch.FloatTensor(wav_samples).to(infer.device)
            #     with torch.no_grad():
            #         mean_scale_latent = infer.generator.extract_latents(wav.unsqueeze(1))

            #     mean, logs_scale = mean_scale_latent.transpose(1,2).chunk(2, dim=2)
            #     audio_latents = torch.randn_like(mean) * torch.exp(logs_scale) + mean

            #     prompt_wav,_ = librosa.load(prompt_wav, sr=self.h.sampling_rate, mono=True)
            #     norm_wav = librosa.util.normalize(prompt_wav) * 0.95
            #     norm_wav = torch.from_numpy(norm_wav)
            #     mel = extract_mel_spec(norm_wav).unsqueeze(0)
            #     mel = modify_vector(mel)

                
            #     prompt_text_tokenized = []
            # else:
            #     prompt_text_tokenized = []
            #     target_text_tokenized = self.tokenizer.encode(target_text)
            #     audio_latents = None


            text_ids = torch.from_numpy(np.asarray( prompt_text_tokenized + target_text_tokenized )).long()
            input_ids = text_ids.to(self.device)
            mel = mel.to(self.device) if mel is not None else None

            input_ids = torch.nn.functional.pad(input_ids,(0,500-input_ids.shape[0]),mode='constant',value=self.tokenizer.pad_token_id)

            with torch.no_grad():
                with torch.autocast(device_type="cuda"):
                    generate_audio_latents = self.model.infer(input_ids,audio_latents,mel,
                                                              bos_token=torch.tensor(self.tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_START|>'),dtype=torch.long,device=self.device),
                                                              txt_pad_token=self.tokenizer.pad_token_id)
                    audio = self.generator.inference_from_latents(generate_audio_latents, do_sample=True) * 32767.0

            audio = audio.detach().cpu().numpy().astype('int16')
            write(os.path.join(self.target_name,f"{utt}.wav"),self.h.sampling_rate,audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-c','--config', type=str, default='./configs/vae_llama_online_node61_sft.yaml', help='Config yaml')
    parser.add_argument('-d','--device', type=str, default='0', help='Device')
    parser.add_argument('-m','--test_mod', type=str, default='en', help='Test mod')
    parser.add_argument('-p','--check_point_path', type=str, default=None, help='Check point path')

    # 选择性
    parser.add_argument('-icl',action='store_true', help='Use prompt')
    parser.add_argument('-stream',action='store_true', help='Use sample')
    parser.add_argument('-useprompt',action='store_true', help='Opt before infer')
    
    args = parser.parse_args()

    infer = infer_tools(args.config, args.device, args.test_mod, args.check_point_path, use_prompt=args.useprompt, streaming=args.stream, icl=args.icl)
    infer.infer()