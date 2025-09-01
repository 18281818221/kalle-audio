import os
import torch
import json
import yaml
import sys
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

if len(sys.argv) > 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]

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

# config = yaml.safe_load(open("./configs/vae_llama_online_node61_sft.yaml"))
config = yaml.safe_load(open("./configs/vae_llama_online_node58.yaml"))
# config = yaml.safe_load(open("./configs/vae_llama_online_node55.yaml"))
config['exp_dir'] = os.path.join(config['exp_dir'],config['project_name'])
config['log_dir'] = os.path.join(config['exp_dir'],'logs')
config['output_dir'] = os.path.join(config['exp_dir'],'output')
config['resume_dir'] = os.path.join(config['exp_dir'],'resume')

if len(sys.argv) > 3:
    check_point_path = sys.argv[3]
else:
    ckpt_dir = os.path.join(config['exp_dir'],'output')
    checkpoints = [os.path.join(config['output_dir'], d) for d in os.listdir(config['output_dir']) if d.startswith("epoch_")]
    check_point_path = max(checkpoints, key=os.path.getmtime)
    # check_point_path = "../epoch_1_step_267599.pt"

if len(sys.argv) > 1:
    test_mod = sys.argv[1]
    if test_mod not in ['en', 'zh']:
        test_mod = 'en'
else:
    test_mod = 'zh'

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

ckpt = torch.load(check_point_path,map_location='cpu')
# model.load_state_dict(ckpt)
# model = model.cuda().eval()
# for name, param in model.named_parameters():
#     assert param.dtype == torch.float32


abs_path = os.path.abspath(check_point_path)
version = os.path.basename(os.path.dirname(os.path.dirname(check_point_path)))
ckpt_name = os.path.basename(check_point_path)
target_name = f"test-{version}-{ckpt_name}-{test_mod}"
os.makedirs(target_name, exist_ok=True)

test_dir = os.path.join('./test_seed_dir', test_mod)
assert os.path.isdir(test_dir)

test_meta = os.path.join(test_dir, 'meta.lst')
meta_lst = open(test_meta).readlines()

for meta in tqdm(meta_lst):
    meta = meta.strip()
    
    utt,prompt_text,prompt_wav,target_text = meta.split('|')

    prompt_text_tokenized = tokenizer.encode(prompt_text)
    target_text_tokenized = tokenizer.encode(target_text)[1:]
    text_ids = torch.from_numpy(np.asarray( prompt_text_tokenized + target_text_tokenized + [speech_understanding_end_id, speech_generation_start_id ] )).long()

    input_ids = text_ids.cuda()

    prompt_wav = os.path.join(test_dir, prompt_wav)
    prompt_wav,_ = librosa.load(prompt_wav, sr=h.sampling_rate, mono=True)
    prompt_wav_tensor = torch.FloatTensor(prompt_wav.reshape(1, -1)).to('cuda')
    with torch.no_grad():
        prompt_mean_scale_latent = generator.extract_latents(prompt_wav_tensor.unsqueeze(0))
    mean, logs_scale = prompt_mean_scale_latent[:,:,:-1].transpose(1,2).chunk(2, dim=2)

    audio_latents = torch.randn_like(mean) * torch.exp(logs_scale) + mean

    model = Llasa(config['model'],tokenizer)
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
            generate_audio_latents = model.infer(input_ids,audio_latents)
            audio = generator.inference_from_latents(generate_audio_latents, do_sample=True) * 32767.0

    audio = audio.detach().cpu().numpy().astype('int16')
    write(os.path.join(target_name,f"{utt}.wav"),h.sampling_rate,audio)