
import os
import sys
import glob
import scipy
import zhconv
import soundfile as sf
from tqdm import tqdm
from funasr import AutoModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration

puncts = [
    '!', ',', '?', '、', '。', '！', '，', '；', '？', '：', '「', '」', '︰', '『', '』',
    '《', '》','.',",", '"', '\'', '?', '-', '—',';'
]

def replace_punctuation_with_space(text):
    
    for punct in puncts:
        text = text.replace(punct, ' ')
    return text


lang = sys.argv[1]
device = f"cuda:{sys.argv[2]}"
wav_dir = sys.argv[3]

def load_en_model():
    model_id = "../iic/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model

def load_zh_model():

    model = AutoModel(
        model="../iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", 
        vad_model="../iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        punc_model="../iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        model_hub="ms",
        disable_update=True,
        disable_pbar=True,
        device=device
    )
    return model

def run_asr(wav_dir):
    if lang == "en":
        processor, model = load_en_model()
    elif lang == "zh":
        model = load_zh_model()

    if 'hard' in wav_dir:
        gt_meta = os.path.join('./test_seed_dir',lang,'hardcase.lst')
        meta_lst = open(gt_meta).readlines()
    else:
        gt_meta = os.path.join('./test_seed_dir',lang,'meta.lst')
        meta_lst = open(gt_meta).readlines()

    res_path = os.path.join(wav_dir,f"aaa_asr.txt")
    gt_path = os.path.join(wav_dir,f"aaa_gt.txt")
    
    fout_asr = open(res_path, "w")
    fout_gt = open(gt_path, "w")
    
    # import pdb;pdb.set_trace()
    for meta in tqdm(meta_lst):
        meta = meta.strip()

        utt,prompt_text,prompt_wav,target_text = meta.split('|')
        wav_path = os.path.join(wav_dir, f"{utt}.wav")

        if lang == "en":
            try:
                wav, sr = sf.read(wav_path)
            except Exception as e:
                continue
            if sr != 16000:
                wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
            input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device)
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
        elif lang == "zh":
            try:
                res = model.generate(input=wav_path,
                        batch_size_s=300)
                transcription = res[0]["text"]
                transcription = zhconv.convert(transcription, 'zh-cn')
            except Exception as e:
                print(e)
                transcription = ""
                
        fout_asr.write(f"{os.path.basename(wav_path)}\t{replace_punctuation_with_space(transcription)}\n")
        fout_gt.write(f"{os.path.basename(wav_path)}\t{replace_punctuation_with_space(target_text)}\n")
        fout_asr.flush()
        fout_gt.flush()

run_asr(wav_dir)