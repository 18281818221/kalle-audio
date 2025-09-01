import torch
import torch.nn.functional as F
import torch.distributions as D
from torch import nn
from transformers import AutoModelForCausalLM

class Llasa(nn.Module):
    def __init__(
        self,
        config,
        tokenizer,
        use_flash_attention = True
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
        self.audio_linear = nn.Linear(config['latent_dim'], config['audio_proj_dim'], dtype=torch.float16) \
                                        if self.use_fa \
                                        else nn.Linear(config['latent_dim'],config['audio_proj_dim'])

        # self.distribution_linear = nn.Linear(config['audio_proj_dim'], config['latent_dim']*2, dtype=torch.bfloat16) \
        #                                 if self.use_fa \
        #                                 else nn.Linear(config['audio_proj_dim'], config['latent_dim']*2 )
        # self.distribution_linear 修改为一个MLP

        self.distribution_linear = nn.Sequential(
            nn.Linear(config['audio_proj_dim'], config['latent_dim'], dtype=torch.float16) \
                                        if self.use_fa \
                                        else nn.Linear(config['audio_proj_dim'], config['latent_dim'] ),
            nn.GELU(),
            nn.Linear(config['latent_dim'], config['latent_dim'], dtype=torch.float16) \
                                        if self.use_fa \
                                        else nn.Linear(config['latent_dim'], config['latent_dim'] ),
        )

        self.init_sigmaVAE()
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
        self.std = self.std.to(input_ids.device)
        text_embed = self.base_model.model.embed_tokens(input_ids)  # b,t,d

        audio_latents = self.sample(mean=audio_latents)
        audio_embed = self.audio_linear(audio_latents)              # b,t,d
        audio_latents_dim = audio_latents.shape[-1]


        input_embed = (audio_embed * audio_mask.unsqueeze(-1)) + (text_embed * ids_mask.unsqueeze(-1)) # b,t,d
        attention_mask = ids_mask + audio_mask
        # import pdb;pdb.set_trace()


        hidden = self.base_model.model(
            inputs_embeds=input_embed,
            attention_mask=attention_mask,
        )[0]  # b,t,d

        x = self.distribution_linear(hidden)       # b,t,d2 

        mean1= audio_distribution_l
        mean2 = x
 
        l_disp = D.Normal(mean1, self.std) # 均值和标准差
        p_disp = D.Normal(mean2, self.std)

        kl = D.kl_divergence(l_disp, p_disp)

        kl = kl.sum(2) / audio_latents_dim
        audio_loss = (kl * target_mask).sum() / target_mask.sum()
        end_loss = (kl * end_mask).sum() / end_mask.sum()

        return {
            "audio_loss": audio_loss,
            "end_loss": end_loss,
            "pre_mean": mean2,
            "pre_log_scale": None
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

            mean,logs_scale2 = last_disp.chunk(2,dim=2)

            
            audio_latent = torch.randn_like(mean) * torch.exp(logs_scale2) + mean

            final_audio_latents_lst.append(last_disp)

            end_disp = D.Normal(torch.ones_like(mean),torch.exp(torch.ones_like(logs_scale2))) # 均值和标准差
            p_disp = D.Normal(mean,torch.exp(logs_scale2))
            latent_dim = mean.shape[2] 
            kl = D.kl_divergence(p_disp, end_disp).sum(2) / latent_dim

            if kl < end_disp_kl_thres and i > 3:
                break

            audio_embed = self.audio_linear(audio_latent)
            input_embed = torch.cat((input_embed,audio_embed),dim=1)

        generate_audio_latents = torch.stack(final_audio_latents_lst[:-1],dim=1).squeeze(1).squeeze(2)
        return generate_audio_latents.transpose(1,2)
    
    def init_sigmaVAE(self):
        self.std = torch.tensor(0.5)
        
    def sample(self, mean, dist_type='fix'):
        """
        Sample from the distribution.
        
        Args:
            dist_type (`str`): Sampling method, either 'fix' or 'gaussian'.
                
        Returns:
            `torch.FloatTensor`: Sampled values.
            `torch.FloatTensor` (optional): Standard deviation used (only when dist_type='gaussian').
        """
        if dist_type == 'fix':
            x = mean + self.std * torch.randn_like(mean)
            return x.to(mean.device)
        elif dist_type == 'gaussian':
            batch_size = mean.size(0)
            value = self.std / 0.8
            std = torch.randn(batch_size, device=mean.device, dtype=mean.dtype) * value

            while std.dim() < mean.dim():
                std = std.unsqueeze(-1)

            x = mean + std * torch.randn_like(mean)
            return x.to(mean.device)
        else:
            return mean

    def kl(self, mean):
        """Compute KL divergence between this distribution and a standard normal."""
        target = torch.zeros_like(mean)
        return F.mse_loss(mean, target, reduction='none')
