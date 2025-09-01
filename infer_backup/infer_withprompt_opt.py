import os
import torch
import json
import yaml
import random
import librosa
import torch.nn as nn
import numpy as np
import torch.distributions as D
import torch.nn.functional as F

# from stable_audio_tools.models import create_model_from_config
from flows import BigVGANFlowVAE as Generator
from torch.optim import AdamW
from transformers import AutoTokenizer,get_scheduler
from scipy.io.wavfile import write
from model import Llasa
from tqdm import tqdm

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

config = yaml.safe_load(open("./configs/vae_llama_online_node58.yaml"))
config['exp_dir'] = os.path.join(config['exp_dir'],config['project_name'])
config['log_dir'] = os.path.join(config['exp_dir'],'logs')
config['output_dir'] = os.path.join(config['exp_dir'],'output')
config['resume_dir'] = os.path.join(config['exp_dir'],'resume')



ckpt_dir = os.path.join(config['exp_dir'],'output')
checkpoints = [os.path.join(config['output_dir'], d) for d in os.listdir(config['output_dir']) if d.startswith("epoch_")]
check_point_path = max(checkpoints, key=os.path.getmtime)

vae_model_config = config['dataset']['vae_config'].get('config_file')
with open(vae_model_config) as f:
    vae_model_config = f.read()

json_config = json.loads(vae_model_config)
h = AttrDict(json_config)
generator = Generator(h)

state_dict_g = load_checkpoint(config['dataset']['vae_config'].get('cpt_path'), 'cpu')
generator.load_state_dict(state_dict_g['generator'])
generator.eval()
generator.remove_weight_norm()
generator.cuda()
torch.backends.cudnn.benchmark = False



tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')    # 128260
speech_generation_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')        # 128261
text_generation_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_START|>')        # 128256
text_generation_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_END|>')            # 128257
text_understanding_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_START|>')  # 128258
text_understanding_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_END|>')      # 128259
speech_understanding_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')  # 128262
speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')      # 128263
 
model = Llasa(config['model'],tokenizer)
latent_dim = config['model']['latent_dim']


ckpt = torch.load(check_point_path)
model.load_state_dict(ckpt)
model = model.cuda()

optimizer = AdamW(
        model.parameters(),
        lr=float(1e-5),
        weight_decay=float(config["weight_decay"]),
    )
scheduler = get_scheduler(
        config["scheduler"],
        optimizer=optimizer,
        num_warmup_steps=15,
        num_training_steps=40,
    )

abs_path = os.path.abspath(check_point_path)
version = os.path.basename(os.path.dirname(os.path.dirname(check_point_path)))
ckpt_name = os.path.basename(check_point_path)
target_name = f"{version}-{ckpt_name}"

# total_lines = open(config['dataset']['meta_path']).readlines()
# prompt_line = random.choice(total_lines)
# prompt_info = json.loads(prompt_line)
# prompt_text = prompt_info['text']
# prompt_wav_path = prompt_info['wav_path']

# prompt_text = "We had mortgage rates at around two percent, close to two percent, and they're now at ten percent, and you can't get a mortgage, so "
# prompt_wav_path = "./trump_2.wav"
prompt_text = "对这就是我万人敬仰的太乙真人虽然有点婴儿肥但也掩不住我逼人的帅气"
prompt_wav_path = "./太乙真人.wav"

prompt_text_tokenized = tokenizer.encode(prompt_text)

prompt_wav,_ = librosa.load(prompt_wav_path, sr=h.sampling_rate, mono=True)
prompt_wav_tensor = torch.FloatTensor(prompt_wav.reshape(1, -1)).to('cuda')
with torch.no_grad():
    prompt_mean_scale_latent = generator.extract_latents(prompt_wav_tensor.unsqueeze(0))
    # prompt_mean_scale_latent -> (b d t)
mean, logs_scale = prompt_mean_scale_latent[:,:,:-1].transpose(1,2).chunk(2, dim=2)

# test_text = '"Uh, are you sure about this?" Tim asked nervously, looking at the steep slope before them. "Whoa, it’s higher than I thought," he continued, his voice filled with trepidation. "Aha, but look at the view," Emily responded with excitement, "it’s worth the climb!"'
# test_text = 'I pity you.'
# test_text = "The princess did not appear to understand him, for she retorted his question"
# test_text = "Almost instantly she was forced to the top."
test_text = '突然，身边一阵笑声。我看着他们，意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道,我身上的肉，是为了掩饰我爆棚的魅力，否则，岂不吓坏了你们呢？'
# test_text = "China is a great country."

text_tokenized = tokenizer.encode(test_text)[1:] # the first token is <s>
text_ids = torch.from_numpy(np.asarray( prompt_text_tokenized + text_tokenized + [speech_understanding_end_id, speech_generation_start_id ] )).long()

input_ids = text_ids.cuda()
audio_latents = torch.randn_like(mean) * torch.exp(logs_scale) + mean
audio_latents = audio_latents

kl_lst = []
mean_lst = []
scale_lst = []

for i in range(50):
    with torch.autocast(device_type="cuda"):
        text_embed = model.base_model.model.embed_tokens(input_ids.unsqueeze(0))
        audio_embed = model.audio_linear(audio_latents)
        input_embed = torch.cat((text_embed,audio_embed),dim=1)
        hidden = model.base_model.model(inputs_embeds=input_embed)[0]
        distribution_p = model.distribution_linear(hidden) 
        audio_len = audio_embed.shape[1]
        dis_p = distribution_p[:,-1-audio_len:-1,:]

        mean2,logs_scale2 = dis_p.chunk(2,dim=2)

        l_disp = D.Normal(mean,torch.exp(logs_scale)) 
        p_disp = D.Normal(mean2,torch.exp(logs_scale2))
        # import pdb;pdb.set_trace()

        kl = D.kl_divergence(p_disp, l_disp)
        kl = kl.sum(2) / audio_latents.shape[-1]
        kl = kl.sum() / audio_len

        print(kl.detach().item())
        # if kl.detach().item() < 1:
        #     break

        kl.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()



with torch.no_grad():
    with torch.autocast(device_type="cuda"):
        text_embed = model.base_model.model.embed_tokens(input_ids.unsqueeze(0))
        audio_embed = model.audio_linear(audio_latents) if audio_latents is not None else None

        input_embed = torch.cat((text_embed,audio_embed),dim=1) if audio_latents is not None else text_embed
        final_audio_latents_lst = []
        for i in tqdm(range(300)):
    
            hidden = model.base_model.model(inputs_embeds=input_embed)
            last_hidden = hidden[0][:,-1:,:]
            last_disp = model.distribution_linear(last_hidden)

            mean,logs_scale2 = last_disp.chunk(2,dim=2)
            # stdev = nn.functional.softplus(scale) + 1e-4
            # audio_latent = torch.randn_like(mean) * torch.exp(logs_scale2) + mean
            temperature = 1 # 大于1的值会增加随机性
            audio_latent = torch.randn_like(mean) * torch.exp(logs_scale2) * temperature + mean

            # final_audio_latents = torch.cat((final_audio_latents,audio_latent),dim=1)
            final_audio_latents_lst.append(last_disp)

            end_disp = D.Normal(torch.ones_like(mean),F.softplus(torch.ones_like(logs_scale2))+ 1e-4 ) # 均值和标准差
            p_disp = D.Normal(mean,torch.exp(logs_scale2))
            kl = D.kl_divergence(p_disp, end_disp).sum(2) / latent_dim
            if kl < 0.5:
                print(kl)
                break
            kl_lst.append(kl)

            audio_embed = model.audio_linear(audio_latent)
            input_embed = torch.cat((input_embed,audio_embed),dim=1)

            
        generate_audio_latents = torch.stack(final_audio_latents_lst,dim=1).squeeze(1).squeeze(2)
        
        # generate_audio_latents = torch.cat((audio_latents,generate_audio_latents),dim=1) if audio_latents is not None else generate_audio_latents
        # import pdb;pdb.set_trace()
        audio = generator.inference_from_latents(generate_audio_latents.transpose(1,2), do_sample=True) * 32767.0
        audio = audio.detach().cpu().numpy().astype('int16')


np.save(f'./{target_name}.npy',generate_audio_latents.detach().cpu().numpy())
write(f'./{target_name}.wav',h.sampling_rate,audio)
write(f"./{target_name}_prompt.wav",h.sampling_rate,prompt_wav)
audio,_ = librosa.load(f"./{target_name}.wav",sr=h.sampling_rate,mono=True)
prompt_plus_audio = np.concatenate((prompt_wav,audio),axis=0)
write(f"./{target_name}_prompt_plus_audio.wav",h.sampling_rate,prompt_plus_audio)

for i ,kl in enumerate(kl_lst):
    print(i,kl)

# for i, mean in enumerate(mean_lst):
#     print(i,mean.std())


