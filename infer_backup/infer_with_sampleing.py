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
from transformers import AutoTokenizer
from scipy.io.wavfile import write
from model import Llasa
from norm_sample import sample_within_confidence_interval
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
# check_point_path = "../epoch_2_step_545000.pt"

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
model = model.cuda().eval()

abs_path = os.path.abspath(check_point_path)
version = os.path.basename(os.path.dirname(os.path.dirname(check_point_path)))
ckpt_name = os.path.basename(check_point_path)
target_name = f"{version}-{ckpt_name}"

total_lines = open('./lirbitts_wav_train.jsonl').readlines()
prompt_line = random.choice(total_lines)
prompt_info = json.loads(prompt_line)
prompt_text = prompt_info['text']
prompt_wav_path = prompt_info['wav_path']

prompt_text = "We had mortgage rates at around two percent, close to two percent, and they're now at ten percent, and you can't get a mortgage, so "
prompt_wav_path = "./trump_2.wav"
# prompt_text = "老公永远状况外，老婆有话大声说。"
# prompt_wav_path = "/home/work_nfs16/kxxia/work/acc_llasa/test_seed_dir/zh/prompt-wavs/00004528-00000113.wav"

# prompt_text = "对这就是我万人敬仰的太乙真人虽然有点婴儿肥但也掩不住我逼人的帅气"
# prompt_wav_path = "./太乙真人.wav"


prompt_text_tokenized = tokenizer.encode(prompt_text)

prompt_wav,_ = librosa.load(prompt_wav_path, sr=h.sampling_rate, mono=True)
prompt_wav_tensor = torch.FloatTensor(prompt_wav.reshape(1, -1)).to('cuda')
with torch.no_grad():
    prompt_mean_scale_latent = generator.extract_latents(prompt_wav_tensor.unsqueeze(0))
    prompt_copysyn = generator.inference_from_latents(prompt_mean_scale_latent[:,:,:-1], do_sample=True) * 32767.0
    prompt_copysyn = prompt_copysyn.detach().cpu().numpy().astype('int16')
    write(f'./prompt_copysyn.wav',h.sampling_rate,prompt_copysyn)

mean, logs_scale = prompt_mean_scale_latent.chunk(2, dim=1)

# test_text = '"Uh, are you sure about this?" Tim asked nervously, looking at the steep slope before them.'
# test_text = 'I pity you.'
# test_text = "The princess did not appear to understand him, for she retorted his question"
test_text = "Almost instantly she was forced to the top."
# test_text = '突然，身边一阵笑声。我看着他们，意气风发地挺直了胸膛，甩了甩那稍显肉感的双臂，轻笑道："我身上的肉，是为了掩饰我爆棚的魅力，否则，岂不吓坏了你们呢？"'
# test_text = "某个倒霉的家伙穿越到了几万年前的斗气大陆，为了保证自己不会和邪族同归于尽，打起持久战的同时，将自己搓出来的帝丹扔到了几万年后的斗气大陆。"
# test_text = "上周末有一天，他回家很晚，早过了他该回来的时候。"
# test_text = "China is a great country."
# test_text = "中国是一个伟大的国家。"
# 10002298-00000019|顺风时提高警惕，逆风时笃定前行。|prompt-wavs/10002298-00000001.wav|我们将为全球城市的可持续发展贡献力量。


text_tokenized = tokenizer.encode(test_text)[1:] # the first token is <s>
text_ids = torch.from_numpy(np.asarray( prompt_text_tokenized + text_tokenized + [speech_understanding_end_id, speech_generation_start_id ] )).long()

input_ids = text_ids.cuda()
audio_latents = torch.randn_like(mean) * torch.exp(logs_scale) + mean
audio_latents = audio_latents.transpose(1,2)[:,:-1,:]

kl_lst = []
mean_lst = []
scale_lst = []

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


            # import pdb;pdb.set_trace()
            # stdev = nn.functional.softplus(scale) + 1e-4
            # audio_latent = torch.randn_like(mean) * torch.exp(logs_scale2) + mean
            # temperature = 1# 大于1的值会增加随机性
            # audio_latent = torch.randn_like(mean) * torch.exp(logs_scale2) * temperature + mean
            audio_latent = sample_within_confidence_interval(mean.cpu().numpy().squeeze(),torch.exp(logs_scale2).cpu().numpy().squeeze(),confidence=0.95,n_samples=1)
            audio_latent = torch.from_numpy(audio_latent).to(last_disp.dtype).to(last_disp.device).permute(1, 0).unsqueeze(0)

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


