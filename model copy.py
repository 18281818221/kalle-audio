import torch
import random
import torch.nn.functional as F
import torch.distributions as D
from torch import nn
from transformers import AutoModelForCausalLM
from norm_sample import sample_within_confidence_interval
# from mrte import MRTE
from ecapa_tdnn import ECAPA_TDNN


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
        self.audio_linear = nn.Linear(config['latent_dim'],2048,dtype=torch.bfloat16) if self.use_fa else nn.Linear(config['latent_dim'],2048)
        self.distribution_linear = nn.Linear(2048,config['latent_dim']*2,dtype=torch.bfloat16) if self.use_fa else nn.Linear(2048,config['latent_dim']*2)
        # self.mrte = MRTE()
        # if self.use_fa:
        #     self.mrte = self.mrte.to(torch.bfloat16)
        self.speaker_encoder = ECAPA_TDNN(in_channels=80, channels=512, embd_dim=2048)
        if self.use_fa:
            self.speaker_encoder = self.speaker_encoder.to(torch.bfloat16)

    def forward(
        self,
        input_ids,              # b,t
        audio_latents,          # b,t,d1
        audio_distribution_l,     # b,t,d2
        mels,                   # b,d,t

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

        # speaker_cond,text_embed = self.mrte(mels,text_embed)
        speaker_cond = self.speaker_encoder(mels.transpose(1,2))

        input_embed = (audio_embed * audio_mask.unsqueeze(-1)) + (text_embed * ids_mask.unsqueeze(-1))   # b,t,d
        attention_mask = ids_mask + audio_mask
        # import pdb;pdb.set_trace()
        input_embed = torch.concat((speaker_cond.unsqueeze(1),input_embed),dim=1)

        true_column = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
        # import pdb;pdb.set_trace()
        attention_mask = torch.cat((true_column,attention_mask ), dim=1)

        hidden = self.base_model.model(
            inputs_embeds=input_embed,
            attention_mask=attention_mask,
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
        mels,                   # b,d,t
        end_disp_kl_thres = 0.5,
        max_length = 1000,
        sample = False,
        use_cfg = None,
        flow = None
    ):
        text_embed = self.base_model.model.embed_tokens(input_ids.unsqueeze(0))
        audio_embed = self.audio_linear(audio_latents) if audio_latents is not None else None
        speaker_cond,text_embed = self.mrte(mels,text_embed)

        input_embed = torch.cat((text_embed,audio_embed),dim=1) if audio_latents is not None else text_embed
        input_embed = torch.concat((speaker_cond.unsqueeze(1),input_embed),dim=1)
        final_audio_latents_lst = []

        if use_cfg == 'v1':
            cfg_attation_mask_text = torch.zeros(1,input_ids.shape[0],dtype=torch.bool)
            cfg_attation_mask_audio = torch.ones(1,audio_latents.shape[1],dtype=torch.bool) if audio_latents is not None else None
        elif use_cfg == 'v2':
            cfg_attation_mask_text = torch.ones(1,input_ids.shape[0],dtype=torch.bool)
            if audio_latents is not None:
                # for j in range(s,e):
                # if random.random() < self.cfg_prob:
                #     audio_latents_mask[i,j] = False
                cfg_attation_mask_audio = torch.ones(1,audio_latents.shape[1],dtype=torch.bool)
                for j in range(audio_latents.shape[1]):
                    if random.random() < self.cfg_prob:
                        cfg_attation_mask_audio[0,j] = False
            else:
                cfg_attation_mask_audio = None
        
        if use_cfg is not None:
            assert not sample
            cfg_one_mask = torch.ones(1,1,dtype=torch.bool).to(text_embed.device)
            cfg_zero_mask = torch.zeros(1,1,dtype=torch.bool).to(text_embed.device)
            cfg_attention_mask = torch.cat((cfg_attation_mask_text,cfg_attation_mask_audio),dim=1) if audio_latents is not None else cfg_attation_mask_text
            cfg_attention_mask = cfg_attention_mask.to(text_embed.device)

        for i in range(max_length):
            hidden = self.base_model.model(inputs_embeds=input_embed)
            last_hidden = hidden[0][:,-1:,:]
            last_disp = self.distribution_linear(last_hidden)

            mean,logs_scale2 = last_disp.chunk(2,dim=2)
            # stdev = nn.functional.softplus(scale) + 1e-4
            # audio_latent = torch.randn_like(mean) * torch.exp(logs_scale2) + mean
            if use_cfg is not None:
                # import pdb;pdb.set_trace()
                cfg_hidden = self.base_model.model(inputs_embeds=input_embed,attention_mask=cfg_attention_mask)
                cfg_last_hidden = cfg_hidden[0][:,-1:,:]
                cfg_last_disp = self.distribution_linear(cfg_last_hidden)

                cfg_mean,cfg_logs_scale2 = cfg_last_disp.chunk(2,dim=2)


            if sample:
                audio_latent = sample_within_confidence_interval(mean.cpu().numpy().squeeze(),torch.exp(logs_scale2).cpu().numpy().squeeze(),confidence=0.95,n_samples=1)
                audio_latent = torch.from_numpy(audio_latent).to(last_disp.dtype).to(last_disp.device).permute(1, 0).unsqueeze(0)
            elif use_cfg is not None:
                # import pdb;pdb.set_trace()
                assert 0
                # audio_latent = batch_weighted_difference_sampling(mean.cpu().numpy().squeeze(),torch.exp(logs_scale2).cpu().numpy().squeeze(),cfg_mean.cpu().numpy().squeeze(),torch.exp(cfg_logs_scale2).cpu().numpy().squeeze(),K=0.1,use_parallel=True)
                # audio_latent = torch.from_numpy(audio_latent).to(last_disp.dtype).to(last_disp.device).unsqueeze(0).unsqueeze(0)
            else:
                audio_latent = torch.randn_like(mean) * torch.exp(logs_scale2) + mean

            if flow is not None:
                audio_latent = audio_latent.transpose(1,2)
                mask = torch.ones([audio_latent.size(0), 1, audio_latent.size(-1)]).to(audio_latent.device)
                with torch.no_grad():
                    audio_latent = flow(audio_latent, mask, reverse=True).transpose(1,2)

            final_audio_latents_lst.append(last_disp)

            end_disp = D.Normal(torch.ones_like(mean),torch.exp(torch.ones_like(logs_scale2))) # 均值和标准差
            p_disp = D.Normal(mean,torch.exp(logs_scale2))
            latent_dim = mean.shape[2] 
            assert latent_dim == 256
            kl = D.kl_divergence(p_disp, end_disp).sum(2) / latent_dim

            if kl < end_disp_kl_thres and i > 3:
                break

            audio_embed = self.audio_linear(audio_latent)
            input_embed = torch.cat((input_embed,audio_embed),dim=1)
            # import pdb;pdb.set_trace()
            if use_cfg == 'v1':
                cfg_attention_mask = torch.cat((cfg_attention_mask,cfg_one_mask),dim=1)
            elif use_cfg == 'v2':
                cfg_attention_mask = torch.cat((cfg_attention_mask,cfg_zero_mask),dim=1)
            
        generate_audio_latents = torch.stack(final_audio_latents_lst[:-1],dim=1).squeeze(1).squeeze(2)
        return generate_audio_latents.transpose(1,2)

class Llasa_text_stream(nn.Module):
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
        if self.use_fa:
            self.speaker_encoder = self.speaker_encoder.to(torch.bfloat16)

    def forward(
        self,
        input_ids,              # b,t
        audio_latents,          # b,t,d1
        audio_distribution_l,   # b,t,d2
        mels,                   # b,d,t

    ):
        # text embedding
        # import pdb;pdb.set_trace()
        text_embed = self.base_model.model.embed_tokens(input_ids)  # b,t,d
        audio_embed = self.audio_linear(audio_latents)              # b,t,d
        audio_latents_dim = audio_latents.shape[-1]

        speaker_cond = self.speaker_encoder(mels.transpose(1,2))

        input_embed = text_embed + audio_embed
        # import pdb;pdb.set_trace()
        input_embed = torch.concat((speaker_cond.unsqueeze(1),input_embed),dim=1)

        # true_column = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
        # import pdb;pdb.set_trace()
        # attention_mask = torch.cat((true_column,attention_mask ), dim=1)

        hidden = self.base_model.model(
            inputs_embeds=input_embed,
            # attention_mask=attention_mask,
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
        audio_loss = kl.mean()
        # audio_loss = (kl * target_mask).sum() / target_mask.sum()
        # end_loss = (kl * end_mask).sum() / end_mask.sum()

        return {
            "audio_loss": audio_loss,
            "end_loss": None,
            "pre_mean": mean2,
            "pre_log_scale": logs_scale2
        }
    
    
    @torch.no_grad()
    def infer(
        self,
        input_ids,              # t
        audio_latents,          # 
        mels,                   # b,d,t

        max_length = 1000,

    ):
        text_embed = self.base_model.model.embed_tokens(input_ids.unsqueeze(0))
        audio_embed = self.audio_linear(audio_latents) 
        speaker_cond = self.speaker_encoder(mels.transpose(1,2))

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
            audio_latents = torch.cat((audio_latents,gen_audio_latent),dim=1)

            final_audio_latents_lst.append(last_disp)


            audio_embed = self.audio_linear(audio_latents)
            input_embed = text_embed[:,:i,:] + audio_embed
           
        generate_audio_latents = torch.stack(final_audio_latents_lst[:-1],dim=1).squeeze(1).squeeze(2)
        return generate_audio_latents.transpose(1,2)


class Llasa_random_drop_spkcond(nn.Module):
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
        if self.use_fa:
            self.speaker_encoder = self.speaker_encoder.to(torch.bfloat16)

    def forward(
        self,
        input_ids,              # b,t
        audio_latents,          # b,t,d1
        audio_distribution_l,   # b,t,d2
        mels,                   # b,d,t
        speaker_cond_keep,      # b

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

        # speaker_cond,text_embed = self.mrte(mels,text_embed)
        speaker_cond = self.speaker_encoder(mels.transpose(1,2))

        speaker_cond = torch.where(speaker_cond_keep.unsqueeze(1), speaker_cond, torch.ones_like(speaker_cond))

        input_embed = (audio_embed * audio_mask.unsqueeze(-1)) + (text_embed * ids_mask.unsqueeze(-1))   # b,t,d
        attention_mask = ids_mask + audio_mask
        # import pdb;pdb.set_trace()
        input_embed = torch.concat((speaker_cond.unsqueeze(1),input_embed),dim=1)

        true_column = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
        # import pdb;pdb.set_trace()
        attention_mask = torch.cat((true_column,attention_mask ), dim=1)

        hidden = self.base_model.model(
            inputs_embeds=input_embed,
            attention_mask=attention_mask,
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
        mels,                   # b,d,t
        end_disp_kl_thres = 0.5,
        max_length = 1000,
        sample = False,
        use_cfg = None,
        flow = None
    ):
        text_embed = self.base_model.model.embed_tokens(input_ids.unsqueeze(0))
        audio_embed = self.audio_linear(audio_latents) if audio_latents is not None else None
        speaker_cond,text_embed = self.mrte(mels,text_embed)

        input_embed = torch.cat((text_embed,audio_embed),dim=1) if audio_latents is not None else text_embed
        input_embed = torch.concat((speaker_cond.unsqueeze(1),input_embed),dim=1)
        final_audio_latents_lst = []

        if use_cfg == 'v1':
            cfg_attation_mask_text = torch.zeros(1,input_ids.shape[0],dtype=torch.bool)
            cfg_attation_mask_audio = torch.ones(1,audio_latents.shape[1],dtype=torch.bool) if audio_latents is not None else None
        elif use_cfg == 'v2':
            cfg_attation_mask_text = torch.ones(1,input_ids.shape[0],dtype=torch.bool)
            if audio_latents is not None:
                # for j in range(s,e):
                # if random.random() < self.cfg_prob:
                #     audio_latents_mask[i,j] = False
                cfg_attation_mask_audio = torch.ones(1,audio_latents.shape[1],dtype=torch.bool)
                for j in range(audio_latents.shape[1]):
                    if random.random() < self.cfg_prob:
                        cfg_attation_mask_audio[0,j] = False
            else:
                cfg_attation_mask_audio = None
        
        if use_cfg is not None:
            assert not sample
            cfg_one_mask = torch.ones(1,1,dtype=torch.bool).to(text_embed.device)
            cfg_zero_mask = torch.zeros(1,1,dtype=torch.bool).to(text_embed.device)
            cfg_attention_mask = torch.cat((cfg_attation_mask_text,cfg_attation_mask_audio),dim=1) if audio_latents is not None else cfg_attation_mask_text
            cfg_attention_mask = cfg_attention_mask.to(text_embed.device)

        for i in range(max_length):
            hidden = self.base_model.model(inputs_embeds=input_embed)
            last_hidden = hidden[0][:,-1:,:]
            last_disp = self.distribution_linear(last_hidden)

            mean,logs_scale2 = last_disp.chunk(2,dim=2)
            # stdev = nn.functional.softplus(scale) + 1e-4
            # audio_latent = torch.randn_like(mean) * torch.exp(logs_scale2) + mean
            if use_cfg is not None:
                # import pdb;pdb.set_trace()
                cfg_hidden = self.base_model.model(inputs_embeds=input_embed,attention_mask=cfg_attention_mask)
                cfg_last_hidden = cfg_hidden[0][:,-1:,:]
                cfg_last_disp = self.distribution_linear(cfg_last_hidden)

                cfg_mean,cfg_logs_scale2 = cfg_last_disp.chunk(2,dim=2)


            if sample:
                audio_latent = sample_within_confidence_interval(mean.cpu().numpy().squeeze(),torch.exp(logs_scale2).cpu().numpy().squeeze(),confidence=0.95,n_samples=1)
                audio_latent = torch.from_numpy(audio_latent).to(last_disp.dtype).to(last_disp.device).permute(1, 0).unsqueeze(0)
            elif use_cfg is not None:
                # import pdb;pdb.set_trace()
                assert 0
                # audio_latent = batch_weighted_difference_sampling(mean.cpu().numpy().squeeze(),torch.exp(logs_scale2).cpu().numpy().squeeze(),cfg_mean.cpu().numpy().squeeze(),torch.exp(cfg_logs_scale2).cpu().numpy().squeeze(),K=0.1,use_parallel=True)
                # audio_latent = torch.from_numpy(audio_latent).to(last_disp.dtype).to(last_disp.device).unsqueeze(0).unsqueeze(0)
            else:
                audio_latent = torch.randn_like(mean) * torch.exp(logs_scale2) + mean

            if flow is not None:
                audio_latent = audio_latent.transpose(1,2)
                mask = torch.ones([audio_latent.size(0), 1, audio_latent.size(-1)]).to(audio_latent.device)
                with torch.no_grad():
                    audio_latent = flow(audio_latent, mask, reverse=True).transpose(1,2)

            final_audio_latents_lst.append(last_disp)

            end_disp = D.Normal(torch.ones_like(mean),torch.exp(torch.ones_like(logs_scale2))) # 均值和标准差
            p_disp = D.Normal(mean,torch.exp(logs_scale2))
            latent_dim = mean.shape[2] 
            assert latent_dim == 256
            kl = D.kl_divergence(p_disp, end_disp).sum(2) / latent_dim

            if kl < end_disp_kl_thres and i > 3:
                break

            audio_embed = self.audio_linear(audio_latent)
            input_embed = torch.cat((input_embed,audio_embed),dim=1)
            # import pdb;pdb.set_trace()
            if use_cfg == 'v1':
                cfg_attention_mask = torch.cat((cfg_attention_mask,cfg_one_mask),dim=1)
            elif use_cfg == 'v2':
                cfg_attention_mask = torch.cat((cfg_attention_mask,cfg_zero_mask),dim=1)
            
        generate_audio_latents = torch.stack(final_audio_latents_lst[:-1],dim=1).squeeze(1).squeeze(2)
        return generate_audio_latents.transpose(1,2)

