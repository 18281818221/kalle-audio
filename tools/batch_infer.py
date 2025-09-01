import os
import torch
import json
import torch.nn as nn
import numpy as np
import torch.distributions as D
import torch.nn.functional as F

# from stable_audio_tools.models import create_model_from_config
from flows import BigVGANFlowVAE as Generator
from transformers import AutoTokenizer
from scipy.io.wavfile import write
from model import Llasa
from tqdm import tqdm

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

vae_model_config = "./configs/melvae/config.json"
with open(vae_model_config) as f:
    vae_model_config = f.read()

json_config = json.loads(vae_model_config)
h = AttrDict(json_config)
generator = Generator(h)

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict
state_dict_g = load_checkpoint("../g_03050000", 'cpu')
generator.load_state_dict(state_dict_g['generator'])
generator.eval()
generator.remove_weight_norm()
generator.cuda()
torch.backends.cudnn.benchmark = False

tokenizer = AutoTokenizer.from_pretrained("./tokenizer_dir")
model = Llasa({'llm_model_name_or_path':"../Llama-3.2-1B-Instruct"},tokenizer)

check_point_path = "/home/node58_tmpdata2/kxxia/exp/vae_tts_without_endloss_v3_3_node58/output/epoch_22_step_5422.pt"

ckpt = torch.load(check_point_path)
model.load_state_dict(ckpt)
model = model.cuda().eval()

abs_path = os.path.abspath(check_point_path)
version = os.path.basename(os.path.dirname(os.path.dirname(check_point_path)))
ckpt_name = os.path.basename(check_point_path)
target_name = f"{version}-{ckpt_name}"

prom_text = 'I pity you.'
mean_scale_latent = torch.from_numpy(np.load('/home/node58_tmpdata2/kxxia/libritts_mean_scale_latent/150_126107_000016_000001.npy'))
mean, scale = mean_scale_latent.chunk(2, dim=1)

stdev = nn.functional.softplus(scale) + 1e-4
latents = torch.randn_like(mean) * stdev + mean

prom_latents = latents.transpose(1,2)


# test_text = '"Uh, are you sure about this?" Tim asked nervously, looking at the steep slope before them. "Whoa, it’s higher than I thought," he continued, his voice filled with trepidation. "Aha, but look at the view," Emily responded with excitement, "it’s worth the climb!"'
# test_text = 'I pity you.'

# test_text = "The princess did not appear to understand him, for she retorted his question"
test_text = "Almost instantly she was forced to the top."
chat = [
            {"role": "user", "content": f"Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{test_text}<|SPEECH_UNDERSTANDING_END|>"},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
        ]
ids = tokenizer.apply_chat_template(chat, tokenize=True)
input_ids = torch.tensor(ids, dtype=torch.long).cuda()
# audio_latents = prom_latents.cuda()[:,:,:]
audio_latents = None

kl_lst = []
mean_lst = []
scale_lst = []

with torch.no_grad():
    with torch.autocast(device_type="cuda"):
        text_embed = model.base_model.model.embed_tokens(input_ids.unsqueeze(0))
        audio_embed = model.audio_linear(audio_latents) if audio_latents is not None else None

        input_embed = torch.cat((text_embed,audio_embed),dim=1) if audio_latents is not None else text_embed
        input_embed = input_embed.repeat(5, 1, 1)
        final_audio_latents_lst = []
        for i in tqdm(range(50)):
    
            hidden = model.base_model.model(inputs_embeds=input_embed)
            last_hidden = hidden[0][:,-1:,:]
            last_disp = model.distribution_linear(last_hidden)

            mean,logs_scale2 = last_disp.chunk(2,dim=2)
            # stdev = nn.functional.softplus(scale) + 1e-4
            audio_latent = torch.randn_like(mean) * torch.exp(logs_scale2) + mean

            # final_audio_latents = torch.cat((final_audio_latents,audio_latent),dim=1)
            final_audio_latents_lst.append(last_disp)

            
            end_disp = D.Normal(torch.ones_like(mean),F.softplus(torch.ones_like(logs_scale2))+ 1e-4 ) # 均值和标准差
            p_disp = D.Normal(mean,torch.exp(logs_scale2))
            kl = D.kl_divergence(p_disp, end_disp).sum(2) / 64
            # if kl < 0.5:
            #     print(kl)
            #     break
            kl_lst.append(kl)
            mean_lst.append(mean)
            scale_lst.append(torch.exp(logs_scale2))

            audio_embed = model.audio_linear(audio_latent)
            input_embed = torch.cat((input_embed,audio_embed),dim=1)

            
        generate_audio_latents = torch.stack(final_audio_latents_lst,dim=1).squeeze(1).squeeze(2)
        # import pdb;pdb.set_trace()
        # generate_audio_latents = torch.cat((audio_latents,generate_audio_latents),dim=1) if audio_latents is not None else generate_audio_latents
        # import pdb;pdb.set_trace()
        audio = generator.inference_from_latents(generate_audio_latents.transpose(1,2), do_sample=True) * 32767.0
        audio = audio.detach().cpu().numpy().astype('int16')

np.save(f'./{target_name}.npy',generate_audio_latents.detach().cpu().numpy())
for i in range(audio.shape[0]):
    write(f'./{target_name}_{i}.wav',h.sampling_rate,audio[i:i+1,:,:])

for i ,kl in enumerate(kl_lst):
    print(i,kl)

# for i, mean in enumerate(mean_lst):
#     print(i,mean.std())


