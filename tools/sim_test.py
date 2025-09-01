# /home/node57_data2/kxxia/env/anaconda3/envs/acc/bin/python
# python sim_test.py zh 0 /home/work_nfs16/kxxia/work/acc_llasa/test-vae_flow_tts_offline-flow_v2-epoch_17_step_73890.pt-zh-prompt
import os
import sys
import yaml
import soundfile as sf
import torch
from collections import defaultdict 
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from torchaudio.transforms import Resample
import glob
from test_seed_dir.models.ecapa_tdnn import ECAPA_TDNN_SMALL


lang = sys.argv[1]
device = f"cuda:{sys.argv[2]}"
wav_dir = sys.argv[3]



MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]


def init_model(model_name, checkpoint=None):
    if model_name == 'unispeech_sat':
        config_path = 'config/unispeech_sat.th'
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='unispeech_sat', config_path=config_path)
    elif model_name == 'wavlm_base_plus':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=config_path)
    elif model_name == 'wavlm_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
    elif model_name == 'hubert_large':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=config_path)
    elif model_name == 'wav2vec2_xlsr':
        config_path = None
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=config_path)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model


def get_emb(model, wav, device='cpu', sample_rate=16000):
    wav, sr = sf.read(wav)

    wav = torch.from_numpy(wav).unsqueeze(0).float().to(device)

    if sr != sample_rate:
        resample = Resample(orig_freq=sr, new_freq=sample_rate).to(device)
        wav = resample(wav)

    with torch.no_grad():
        emb = model(wav)

    return emb



model = init_model('wavlm_large', '/home/work_nfs16/zhguo/code/dualvc/test/smos/ckpt/wavlm_large_finetune.pth')
model.eval()
model.to(device)

if 'hard' in wav_dir:
    gt_meta = os.path.join('./test_seed_dir',lang,'hardcase.lst')
    meta_lst = open(gt_meta).readlines()
else:
    gt_meta = os.path.join('./test_seed_dir',lang,'meta.lst')
    meta_lst = open(gt_meta).readlines()
# gt_meta = os.path.join('./test_seed_dir',lang,'meta.lst')
# meta_lst = open(gt_meta).readlines()

final_map = {}
final_lst = []

for meta in tqdm(meta_lst):
    meta = meta.strip()
    utt,prompt_text,prompt_wav,target_text = meta.split('|')
    prompt_wav = os.path.join('./test_seed_dir',lang, prompt_wav)
    wav_path = os.path.join(wav_dir, f"{utt}.wav")
    try:
        target_spk_emb = get_emb(model, prompt_wav, device, sample_rate=16000)
        spk_emb = get_emb(model, wav_path, device, sample_rate=16000)
    except:
        continue

    sim = F.cosine_similarity(target_spk_emb, spk_emb)
    final_map[utt] = sim.item()
    final_lst.append(sim.item())
    
open(f"{wav_dir}/0000000_sim,json", "w").writelines([f"{k} {v}\n" for k,v in final_map.items()])
open(f"{wav_dir}/0000000_sim.txt", "w").writelines(f"{sum(final_lst) / len(final_lst)}\n" )


    


