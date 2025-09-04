import datetime
print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] start train.py')
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('numba').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
from itertools import cycle
import yaml
import os
import shutil
import time
import torch.multiprocessing
import torch.multiprocessing as mp
import random
import torch.distributions as D
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, get_scheduler, AutoTokenizer
from torch.optim import AdamW
from model_sigmaVAE import Llasa
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from twj_dataset_offline import TTSDataset_online_parquet, dict_to_device
import os
import shutil
import sys
import torchaudio
from einops import rearrange
print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] import end train.py')
yaml_path = sys.argv[1]


sys.path.append('/mnt/bn/twj-data-multimodal2/workspace/VibeVoice/vibevoice')
sys.path.append('/mnt/bn/twj-data-multimodal2/workspace/VibeVoice')

from modular.modeling_vibevoice import VibeVoiceModel, VibeVoiceForConditionalGeneration
from modular.modular_vibevoice_tokenizer import VibeVoiceAcousticTokenizerModel
from modular.configuration_vibevoice import VibeVoiceAcousticTokenizerConfig



def write_log_to_file(log_content, log_path):
    """将日志内容追加写入文本文件"""
    with open(log_path, "a", encoding="utf-8") as f:  # "a"表示追加模式，不会覆盖旧内容
        f.write(log_content + "\n")  # 每行日志后加换行符，保证格式清晰
def main():
    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] start func main')
    config_file = yaml_path
    config = yaml.safe_load(open(config_file))
    config['exp_dir'] = os.path.join(config['exp_dir'],config['project_name'])
    config['log_dir'] = os.path.join(config['exp_dir'],'logs')
    config['output_dir'] = os.path.join(config['exp_dir'],'output')
    config['resume_dir'] = os.path.join(config['exp_dir'],'resume')

    os.makedirs(config['exp_dir'],exist_ok=True)
    os.makedirs(config['log_dir'],exist_ok=True)
    os.makedirs(config['output_dir'],exist_ok=True)
    os.makedirs(config['resume_dir'],exist_ok=True)
    shutil.copyfile(config_file, os.path.join(config['exp_dir'],'config.yaml'))



    # --------------------------
    # 1. 初始化日志文件（建议在训练开始前执行，仅执行一次）
    # --------------------------
    # 定义日志文件路径（可自定义，例如放在输出目录下）
    log_filename = f"train_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file_path = os.path.join(config['exp_dir'], log_filename)
    eval_audio_path = os.path.join(config['exp_dir'], f'eval_audios_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(eval_audio_path, exist_ok=True)



    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]  preparation end')
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=config["log_dir"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"]
    )
    # accelerator.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
    # accelerator.deepspeed_plugin.deepspeed_config['gradient_accumulation_steps'] = config["gradient_accumulation_steps"]
    accelerator.init_trackers(project_name=config["project_name"])

    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] start loading AutoTokenizer')

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])

    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] start loading Llasa')

    model = Llasa(config['model'],tokenizer,use_flash_attention=config['use_flash_attation'])

    epoch = 0
    step = 0
    
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],  # warmup 步数
        num_training_steps=config["total_steps"],  # 总训练步数
    )
    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] start loading TTSDataset_online_parquet')

    base_lst = config["dataset"]["meta_path"]
    base_lst = [ base_lst ]
    random.seed(42)
    train_dataset = TTSDataset_online_parquet(config["dataset"],tokenizer,base_lst,accelerator.device,output_bf16=config['use_flash_attation'])
    # print('dataset size', len(train_dataset))
    print('dataset size', len(train_dataset))
    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] start loading DataLoader')


    last_checkpoint = None
    if os.path.isdir(config['output_dir']) and len(os.listdir(config['output_dir'])) > 0:
        checkpoints = [os.path.join(config['output_dir'], d) for d in os.listdir(config['output_dir']) if d.startswith("epoch_")]
        if len(checkpoints) > 0:
            # Get the most recent checkpoint based on modification time
            last_checkpoint = max(checkpoints, key=os.path.getmtime)
            epoch = int(last_checkpoint.split('/')[-1].split('_')[1])
            step = int(last_checkpoint.split('/')[-1].split('_')[3].split('.')[0])
    elif config.get("start_checkpoint") is not None :
        assert os.path.exists(config.get("start_checkpoint"))
        last_checkpoint = config.get("start_checkpoint")
        print(f"-------------------------Start training from {last_checkpoint}")
    
    if last_checkpoint is not None:
        print(f"-------------------------Loading model and tokenizer from checkpoint {last_checkpoint}")
        model.load_state_dict(torch.load(last_checkpoint,map_location='cpu'),strict=False)

    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] accelerator prepare.....')


    batch_record_lst = []

    # Use PyTorch DataLoader to yield fixed-size batches (static batch size)
    static_batch_size = int(config["batch_generator"].get("batch_size", 1))

    print(f' static_batch_size {static_batch_size}')
    print(f' total_batch_size (all GPUs) {static_batch_size * accelerator.num_processes}')


    # if accelerator.num_processes > 1:
    #     sampler = DistributedSampler(train_dataset)
    # else:
    #     sampler = None
    if accelerator.num_processes > 1:
        print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] DDP DataLoader prepare.....')

        # accelerator框架中不需要sampler
        data_loader = DataLoader(
            train_dataset,
            batch_size=static_batch_size,
            # sampler=sampler,
            num_workers=config['datapool']['num_workers'],
            pin_memory=True,
            collate_fn=train_dataset.collate,
            drop_last=False,  # 分布式模式下关闭
            shuffle=False,
            prefetch_factor=2
        )
    else:
        print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Single Thread DataLoader prepare.....')

        # accelerator框架中不需要sampler
        data_loader = DataLoader(
            train_dataset,
            batch_size=static_batch_size,
            # sampler=sampler,
            num_workers=config['datapool']['num_workers'],
            pin_memory=True,
            collate_fn=train_dataset.collate,
            drop_last=True,
            shuffle=True,
            prefetch_factor=2
        )




    print("Initializing VAE for evaluation...")
    model_ckpt_path = "/mnt/bn/twj-data-multimodal2/workspace/VibeVoice/VibeVoice-1.5B"
    vae_model = VibeVoiceModel.from_pretrained(model_ckpt_path)
    generator = vae_model.acoustic_tokenizer
    generator.to(accelerator.device)
    generator.eval()
    print("VAE initialized.")


    model, optimizer, scheduler, data_loader = accelerator.prepare(
        model, optimizer, scheduler, data_loader
    )

    # 初始化梯度缩放器（放在训练循环外）
    # scaler = torch.cuda.amp.GradScaler()
    audio_loss = None
    total_loss = None
    end_loss = None

    while True:

        # if using DistributedSampler, set epoch for shuffling
        # if sampler is not None:
        #     sampler.set_epoch(epoch)

        for batch in tqdm(data_loader, disable=not accelerator.is_local_main_process):

            # batch is already collated dict from collate_fn
            # move to device
            batch_size_actual = batch.get("input_ids").shape[0] if batch.get("input_ids") is not None else static_batch_size
            batch_record_lst.append(batch_size_actual)

            with accelerator.accumulate(model):

                optimizer.zero_grad()
                # try:
                outputs = model(
                    input_ids=              batch.get("input_ids"),             # batch_size, len
                    audio_latents=          batch.get("audio_latents"),         # batch_size, len, audio_dim
                    audio_distribution_l=   batch.get("distribute_lables"),     # batch_size, len, audio_dim * 2

                ids_mask=               batch.get("text_ids_mask"),
                audio_mask=             batch.get("audio_latents_mask"),
                target_mask=            batch.get("distribute_lables_mask"),
                end_mask=               batch.get("enddist_mask"),

                # mels=                   batch.get("mels"),
                # speaker_cond_keep=      batch.get("speaker_cond_keep"),
                )   

                audio_loss = outputs['audio_loss']
                end_loss = outputs['end_loss']

                total_loss = (
                                audio_loss*config["audio_loss_weight"] 
                                + end_loss*config["end_loss_weight"]
                            #    + kl*config["kl_loss_weight"]
                )

                accelerator.backward(total_loss)
                # gradient clipping
                # if accelerator.sync_gradients:
                    # accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                accelerator.log({"total_loss": total_loss.detach().item()} ,step = step)
                accelerator.log({"audio_loss": audio_loss.detach().item()} ,step = step)
                accelerator.log({"end_loss": end_loss.detach().item()} ,step = step)
                # accelerator.log({"grad_scale": scaler._scale.item()} ,step = step)

            # accelerator.log({"kl_loss": kl.detach().item()} ,step = step)
                # except Exception as e:
                #     print('error', e)

                accelerator.wait_for_everyone()
                    
                if step != 0 and step % config["save_interval"] == 0:
                    if accelerator.is_main_process:
                        torch.save(accelerator.unwrap_model(model).state_dict(),os.path.join(config['output_dir'],f"epoch_{epoch}_step_{step}.pt"))
                    # accelerator.save_state(f"{config['resume_dir']}/step_{step}")

                if step % config["log_interval"] == 0:
                    if accelerator.is_main_process:
                        print(f' input id shape {batch.get("input_ids").shape}')

                        log_content = "{}: Epoch:{}, Step:{}, batch_size:{}, lr:{}, total_loss:{}, audio_loss:{}, end_loss:{}, data_queue_size:{}".format(
                        # accelerator.print("{}: Epoch:{}, Step:{}, batch_size:{}, lr:{}, total_loss:{}, audio_loss:{}, end_loss:{}，kl_loss:{}, data_queue_size:{}".format(
                            time.ctime(),
                            epoch,
                            step,
                            batch_size_actual,
                            optimizer.param_groups[0]['lr'],
                            total_loss.detach().item(),
                            audio_loss.detach().item(),
                            end_loss.detach().item(),
                            # kl.detach().item(),
                            'N/A'
                            ) 
                        accelerator.print(log_content)
                        batch_record_lst = []

                        # 3. 写入文本日志文件（调用上面定义的函数）
                        write_log_to_file(log_content, log_file_path)
                        
# pred generation
                        pred_mean = outputs['pre_mean'][0].unsqueeze(0)
                        target_text = batch['raw_texts'][0]
                        with torch.no_grad():
                            pred_latent = model.sample(mean=pred_mean)  # 1, 129, 64
                        print('pred_latent', pred_latent.shape)
                        if pred_latent.shape[1] == 64:
                            pred_latent=pred_latent.transpose(1, 2)
                        pred_latent = pred_latent[:, audio_mask[0], :]  # 去掉前面的text部分
                        audio = generator.decode(pred_latent)
                        output_audio_file = os.path.join(eval_audio_path, f'sample_{step}-gen.wav')
                        audio = rearrange(audio, "b d n -> d (b n)")
                        sample_rate = 24000
                        audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                        torchaudio.save(output_audio_file, audio, sample_rate)
                        output_txt_file = output_audio_file.replace('.wav', '.txt')
                        with open(output_txt_file, 'w') as f:
                            f.write(target_text)
                        print(f'Eval wrote audio to {output_audio_file} and text to {output_txt_file}')


# GT generation
                        ground_truth_audio_latents = outputs['ground_truth_audio_latents'][0].unsqueeze(0)
                        if ground_truth_audio_latents.shape[1] == 64:
                            ground_truth_audio_latents=ground_truth_audio_latents.transpose(1, 2)
                        ground_truth_audio_latents = ground_truth_audio_latents[:, audio_mask[0], :]  # 去掉前面的text部分
                        audio = generator.decode(ground_truth_audio_latents)
                        output_audio_file = os.path.join(eval_audio_path, f'sample_{step}-gt.wav')
                        audio = rearrange(audio, "b d n -> d (b n)")
                        sample_rate = 24000
                        audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                        torchaudio.save(output_audio_file, audio, sample_rate)
                        output_txt_file = output_audio_file.replace('.wav', '.txt')
                        with open(output_txt_file, 'w') as f:
                            f.write(target_text)

                        
                        shutil.copy2(batch['speech_paths'][0], os.path.join(eval_audio_path, f'sample_{step}-gt2.wav'))

            
            step += 1
        
        epoch += 1
        train_dataset.set_epoch(epoch)


        if accelerator.is_main_process:
            torch.save(accelerator.unwrap_model(model).state_dict(),os.path.join(config['output_dir'],f"epoch_{epoch}_step_{step}.pt"))


if __name__ == '__main__':
    main()
