
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
import torchaudio
import soundfile as sf
from einops import rearrange
from tqdm import tqdm
# from model import Llasa
from flows import BigVGANFlowVAE as Generator
from transformers import AutoTokenizer,get_scheduler
from scipy.io.wavfile import write

class Llasa(nn.Module):
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
        self.audio_linear = nn.Linear(config['latent_dim'], config['audio_proj_dim'], dtype=torch.bfloat16) \
                                        if self.use_fa \
                                        else nn.Linear(config['latent_dim'],config['audio_proj_dim'])
        self.distribution_linear = nn.Sequential(
            nn.Linear(config['audio_proj_dim'], config['latent_dim']*2, dtype=torch.float16) \
                                        if self.use_fa \
                                        else nn.Linear(config['audio_proj_dim'], config['latent_dim']*2 ),
            nn.GELU(),
            nn.Linear(config['latent_dim']*2, config['latent_dim']*2, dtype=torch.float16) \
                                        if self.use_fa \
                                        else nn.Linear(config['latent_dim']*2, config['latent_dim']*2 ),
        )
    def forward(
        self,
        input_ids,              # b,t
        audio_latents,          # b,t,d1
        audio_distribution_l,     # b,t,d2

        ids_mask,
        audio_mask,
        target_mask,
        end_mask,
    ):
        # text embedding
        # import pdb;pdb.set_trace()
        text_embed = self.base_model.model.embed_tokens(input_ids)  # b,t,d
        audio_embed = self.audio_linear(audio_latents)              # b,t,d
        audio_latents_dim = audio_latents.shape[-1]

        input_embed = (audio_embed * audio_mask.unsqueeze(-1)) + (text_embed * ids_mask.unsqueeze(-1))   # b,t,d
        attention_mask = ids_mask + audio_mask

        hidden = self.base_model.model(
            inputs_embeds=input_embed,
            attention_mask=attention_mask,
        )[0]  # b,t,d

        distribution_p = self.distribution_linear(hidden)       # b,t,d2

        mean1, std1 = get_mean_stdev_from_stableaudio2_latents(audio_distribution_l.transpose(1,2))
        mean1 = mean1.transpose(1,2)
        std1 = std1.transpose(1,2)

        mean2, logs_scale2 = distribution_p.chunk(2,dim=2)
        std2 = torch.exp(logs_scale2)


        l_disp = D.Normal(mean1,std1) # 均值和标准差
        p_disp = D.Normal(mean2,std2)

        kl = D.kl_divergence(p_disp, l_disp)

        kl = kl.sum(2) / audio_latents_dim
        audio_loss = (kl * target_mask).sum() / target_mask.sum()
        end_loss = (kl * end_mask).sum() / end_mask.sum()

        return {
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
        end_disp_kl_thres = 0.5,
        max_length = 1000,
        sample = False,
        use_cfg = None,
        flow = None
    ):
        text_embed = self.base_model.model.embed_tokens(input_ids.unsqueeze(0))
        audio_embed = self.audio_linear(audio_latents) if audio_latents is not None else None

        input_embed = torch.cat((text_embed,audio_embed),dim=1) if audio_latents is not None else text_embed
        final_audio_latents_lst = []
        
        for i in range(max_length):
            hidden = self.base_model.model(inputs_embeds=input_embed)
            last_hidden = hidden[0][:,-1:,:]
            last_disp = self.distribution_linear(last_hidden)

            mean, logs_scale2 = last_disp.chunk(2,dim=2)

            
            audio_latent = torch.randn_like(mean) * torch.exp(logs_scale2) + mean

            final_audio_latents_lst.append(last_disp)

            end_disp = D.Normal(torch.ones_like(mean), \
                                torch.exp(torch.ones_like(logs_scale2))) # 均值和标准差
            p_disp = D.Normal(mean, torch.exp(logs_scale2))
            latent_dim = mean.shape[2] 
            kl = D.kl_divergence(p_disp, end_disp).sum(2) / latent_dim

            if kl < end_disp_kl_thres and i > 3:
                break

            audio_embed = self.audio_linear(audio_latent)
            input_embed = torch.cat((input_embed, audio_embed),dim=1)

        generate_audio_latents = torch.stack(final_audio_latents_lst[:-1],dim=1).squeeze(1).squeeze(2)
        return generate_audio_latents.transpose(1,2)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class infer_tools:
    def __init__(self, 
                 config, 
                 device, 
                 test_mod, 
                 check_point_path,
                 use_prompt=False,
                 use_sample=False,
                 opt_before_infer=False,
                 utt2prompt_wav=None,
                 utt2prompt_txt=None,
                 utt2target_txt=None):
        self.init_config(config)
        
        self.test_mod = test_mod
        self.device = f"cuda:{device}"
        self.check_point_path = check_point_path

        self.use_prompt = use_prompt
        self.use_sample = use_sample
        self.opt_before_infer = opt_before_infer

        self.utt2prompt_wav = utt2prompt_wav
        self.utt2prompt_txt = utt2prompt_txt
        self.utt2target_txt = utt2target_txt

        if self.check_point_path is None:
            checkpoints = [os.path.join(self.config['output_dir'], d) for d in os.listdir(self.config['output_dir']) if d.startswith("epoch_")]
            self.check_point_path = max(checkpoints, key=os.path.getmtime)
            print('No check point found, use the latest one')
        print(' use check point path ', self.check_point_path)

        self.init_vae_generator(self.config['dataset'].get('vae_config'))
        self.init_tokenizer()
        self.init_model()
        self.init_output_info()

        

    def init_config(self, config):
        config = yaml.safe_load(open(config))
        self.config = config
        self.config['exp_dir'] = os.path.join(config['exp_dir'],config['project_name'])
        self.config['output_dir'] = os.path.join(config['exp_dir'],'output')


    def init_vae_generator(self, vae_config):

        print(vae_config)

        vae_model_config = vae_config.get('config_file')
        vae_cpkt_path = vae_config.get('cpt_path')

        print('vae_model_config,', vae_model_config)
        print('vae_cpkt_path', vae_cpkt_path)

        with open(vae_model_config) as f:
            vae_model_config = json.load(f)
        from stable_audio_tools.models import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict
        self.sample_rate = vae_model_config["sample_rate"]
        self.generator = create_model_from_config(vae_model_config)

        print(' start loading ckpt from ', vae_cpkt_path)
        ckpt = load_ckpt_state_dict(vae_cpkt_path)
        
        miss, unexpected = self.generator.load_state_dict(ckpt, strict=False)
        print("generator Missing keys:", miss)
        print("generator Unexpected keys:", unexpected)
        
        for param in self.generator.parameters():
            param.requires_grad = False
        self.generator.to(self.device)
        self.generator.eval()
        torch.backends.cudnn.benchmark = False

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
        model = Llasa(self.config['model'],self.tokenizer)
        self.latent_dim = self.config['model']['latent_dim']
        print(' start loading ckpt from ', self.check_point_path)
        self.ckpt = torch.load(self.check_point_path,map_location='cpu')
        model.load_state_dict(self.ckpt)
        for name, param in model.named_parameters():
            assert param.dtype == torch.float32
        self.model = model.to(self.device)
        print(' init model complete ')


    def infer_stableaudio2(self):
        
        info_lines=[
                "/mnt/bn/twj-data-multimodal2/workspace/kalle/data/test.jsonl"

        ]
        print(' use inference data ', info_lines)
        from twj_utils import read_parquet, vae_sample, read_jsonl
        self.info_lines = read_jsonl(info_lines)
        for item in tqdm(self.info_lines):
            target_text = item['caption']
            utt = item['id']
            vae_path = item['vae_latent_path'].replace('/mnt/bn/twj-data-multimodal/twj/dataset/audioset_gemini2flash/data_dir/audio-dataset-000000/', '/mnt/bn/twj-data-multimodal2/workspace/kalle/data/testset/')

            output_file = os.path.join(self.output_dir, f'{utt}.txt')
            with open(output_file, 'w') as f:
                f.write(target_text)

            if os.path.exists(os.path.join(self.output_dir, f"{utt}.wav")):
                continue
            
            # copysyn audio

            latent = np.load(vae_path) # 64, t
            latent = torch.from_numpy(latent).to(self.device)  # 1, 64, t
            if len(latent.shape)==2:
                latent = latent.unsqueeze(0)
            if latent.shape[1] == 128:
                print('latent dim is 128, use samples: ', latent.shape)
                mean, scale = latent.chunk(2, dim=1) 
                latent, kl = vae_sample(mean, scale) 
            else:
                print('latent dim is 64 not use samples: ', latent.shape)
            output = self.generator.pretransform.decode(latent)
            output = rearrange(output, "b d n -> d (b n)")
            output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            output_file = os.path.join(self.output_dir, f'{utt}--copysyn.wav')
            torchaudio.save(output_file, output, self.sample_rate)



            prompt_text_tokenized = []
            target_text_tokenized = self.tokenizer.encode(target_text)
            audio_latents = None
            text_ids = torch.from_numpy(np.asarray( prompt_text_tokenized + target_text_tokenized + [self.speech_understanding_end_id, self.speech_generation_start_id ] )).long()
            input_ids = text_ids.to(self.device)
            print(f'target_text;{target_text}')
            print(f'input_ids : {input_ids.shape}')
            with torch.no_grad():
                with torch.autocast(device_type="cuda"):
                    generate_audio_latents = self.model.infer(input_ids, audio_latents)
                    print(f' generate_audio_latents :{generate_audio_latents.shape}')


                    if len(generate_audio_latents.shape)==2:
                        generate_audio_latents = generate_audio_latents.unsqueeze(0)
                    if generate_audio_latents.shape[1] == 128:
                        print('generate_audio_latents dim is 128, use samples: ', generate_audio_latents.shape)
                        mean, logs_scale2 = generate_audio_latents.chunk(2, dim=1) 
                        std2 = torch.exp(logs_scale2) * 0.8
                        generate_audio_latents = torch.randn_like(mean) * std2 + mean
                    else:
                        print('generate_audio_latents dim is 64 not use samples: ', generate_audio_latents.shape)
                        
                    output = self.generator.pretransform.decode(generate_audio_latents)
                    output = rearrange(output, "b d n -> d (b n)")

            output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            output_file = os.path.join(self.output_dir, f'{utt}--gen.wav')
            torchaudio.save(output_file, output, self.sample_rate)




    def init_output_info(self):
        import datetime

        version = os.path.basename(os.path.dirname(os.path.dirname(self.check_point_path)))
        ckpt_name = os.path.basename(self.check_point_path)
        self.output_dir = f"inference_results/{version}-{ckpt_name}-[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        # if self.use_prompt:
        #     self.output_dir += '-prompt'
        # if self.use_sample:
        #     self.output_dir += '-sample'
        # if self.opt_before_infer:
        #     self.output_dir += '-opt_before_infer'
        os.makedirs(self.output_dir, exist_ok=True)

    def optim(self,
              input_ids,
              prompt_mean_scale_latent,
              loss_threashold=None,#0.8,
              max_step=200,
              warmup_step=60,#
              training_steps=120,#
              lr=1e-6, # 2e-5 可能还和初始kl有关
              ):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,weight_decay=float(self.config["weight_decay"]))
        scheduler = get_scheduler(
            self.config["scheduler"],
            optimizer=optimizer,
            num_warmup_steps=warmup_step,
            num_training_steps=training_steps,
        )
        mean, logs_scale = prompt_mean_scale_latent[:,:,:-1].transpose(1,2).chunk(2, dim=2)
        for step in range(max_step):
            audio_latents = torch.randn_like(mean) * torch.exp(logs_scale) + mean
            with torch.autocast(device_type="cuda"):
                text_embed = self.model.base_model.model.embed_tokens(input_ids.unsqueeze(0))
                audio_embed = self.model.audio_linear(audio_latents)
                input_embed = torch.cat((text_embed,audio_embed),dim=1)
                hidden = self.model.base_model.model(inputs_embeds=input_embed)[0]
                distribution_p = self.model.distribution_linear(hidden) 
                audio_len = audio_embed.shape[1]
                dis_p = distribution_p[:,-1-audio_len:-1,:]

                mean2,logs_scale2 = dis_p.chunk(2,dim=2)

                l_disp = D.Normal(mean,torch.exp(logs_scale)) 
                p_disp = D.Normal(mean2,torch.exp(logs_scale2))
                # import pdb;pdb.set_trace()

                kl = D.kl_divergence(p_disp, l_disp)
                kl = kl.sum(2) / audio_latents.shape[-1]
                kl = kl.sum() / audio_len

                # print(kl.detach().item())
                if loss_threashold is not None and kl.detach().item() < loss_threashold:
                    break
                # if kl.detach().item() < 1:
                #     break

                kl.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # /mnt/bn/twj-data-multimodal2/logdir/kalle/0828-vggsound_audiosetgemini-sigmaVAE/output/epoch_55_step_105000.pt

    parser.add_argument('-c','--config', type=str, default='/mnt/bn/twj-data-multimodal2/workspace/kalle/configs/twj.yaml', help='Config yaml')
    parser.add_argument('-d','--device', type=str, default='0', help='Device')
    parser.add_argument('-m','--test_mod', type=str, default='en', help='Test mod')
    parser.add_argument('-p','--check_point_path', type=str, default="/mnt/bn/twj-data-multimodal2/logdir/kalle/go_on-go_on-go_on-go_on-go_on-vggsound_audiosetgemini/output/epoch_34_step_65000.pt", help='Check point path')

    # 选择性
    parser.add_argument('-i',action='store_true', help='Use prompt')
    parser.add_argument('-s',action='store_true', help='Use sample')
    parser.add_argument('-o',action='store_true', help='Opt before infer')
    
    args = parser.parse_args()


    print(' init infer_tools ')
    infer = infer_tools(args.config, args.device, args.test_mod, args.check_point_path, args.i, args.s, args.o)

    print(' init infer_tools complete ')
    print(' start infer_stableaudio2 ')
    infer.infer_stableaudio2()