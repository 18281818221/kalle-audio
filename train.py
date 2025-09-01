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

from twj_dataset import TTSDataset_online_parquet, dict_to_device
from model import Llasa
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] import end train.py')

def split_file_lst(file_lst,world_size,rank):
    average = len(file_lst)//world_size
    return file_lst[rank*average:(rank+1)*average]


def main():
    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] start func main')
    config_file = "/mnt/bn/twj-data-multimodal2/workspace/kalle/configs/twj.yaml"
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
    scheduler = get_scheduler(
        config["scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["total_steps"],
    )

    print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] start loading TTSDataset_online_parquet')

    base_lst = config["dataset"]["meta_path"]
    base_lst = [ base_lst ]
    random.seed(42)
    # base_lst = split_file_lst(base_lst,accelerator.num_processes,accelerator.process_index)
    train_dataset = TTSDataset_online_parquet(config["dataset"],tokenizer,base_lst,accelerator.device,output_bf16=config['use_flash_attation'])

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
    if accelerator.num_processes > 1:
        sampler = DistributedSampler(train_dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=False)
    else:
        sampler = None

    data_loader = DataLoader(
        train_dataset,
        batch_size=static_batch_size,
        sampler=sampler,
        num_workers=config['datapool']['num_workers'],
        pin_memory=True,
        collate_fn=lambda x: train_dataset.collate(x),
    )
    model, optimizer, scheduler, data_loader = accelerator.prepare(
        model, optimizer, scheduler, data_loader
    )

    # 初始化梯度缩放器（放在训练循环外）
    scaler = torch.cuda.amp.GradScaler()

    while True:

        # if using DistributedSampler, set epoch for shuffling
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch in tqdm(data_loader, disable=not accelerator.is_local_main_process):
            # batch is already collated dict from collate_fn
            # move to device
            batch_size_actual = batch.get("input_ids").shape[0] if batch.get("input_ids") is not None else static_batch_size
            batch_record_lst.append(batch_size_actual)
            # batch = dict_to_device(batch, accelerator.device)
            # import pdb;pdb.set_trace()

            with accelerator.accumulate(model):

                optimizer.zero_grad()
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

                print('total_loss:', total_loss.item())
                
                # 使用scaler缩放损失并反向传播
                scaler.scale(total_loss).backward()
                print('scaler.scale(total_loss).backward()')
                
                # 使用scaler更新优化器
                scaler.step(optimizer)
                print('scaler.step(optimizer)')
                
                # 更新缩放器状态
                scaler.update()
                print('scaler.update()')
                print('scaler._scale.item()', scaler._scale.item())
                
                scheduler.step()
                print('scheduler')

            accelerator.log({"total_loss": total_loss.detach().item()} ,step = step)
            accelerator.log({"audio_loss": audio_loss.detach().item()} ,step = step)
            accelerator.log({"end_loss": end_loss.detach().item()} ,step = step)
            accelerator.log({"grad_scale": scaler._scale.item()} ,step = step)

            # accelerator.log({"kl_loss": kl.detach().item()} ,step = step)

            if step != 0 and step % config["save_interval"] == 0:
                if accelerator.is_main_process:
                    torch.save(accelerator.unwrap_model(model).state_dict(),os.path.join(config['output_dir'],f"epoch_{epoch}_step_{step}.pt"))
                # accelerator.save_state(f"{config['resume_dir']}/step_{step}")

            if step % 50 == 0:
                accelerator.print("{}: Epoch:{}, Step:{}, batch_size:{}, lr:{}, total_loss:{}, audio_loss:{}, end_loss:{}, data_queue_size:{}".format(
                # accelerator.print("{}: Epoch:{}, Step:{}, batch_size:{}, lr:{}, total_loss:{}, audio_loss:{}, end_loss:{}，kl_loss:{}, data_queue_size:{}".format(
                    time.ctime(),
                    epoch,
                    step,
                    sum(batch_record_lst) / len(batch_record_lst),
                    optimizer.param_groups[0]['lr'],
                    total_loss.detach().item(),
                    audio_loss.detach().item(),
                    end_loss.detach().item(),
                    # kl.detach().item(),
                    'N/A'
                    ) )
                batch_record_lst = []

            step += 1
        
        epoch += 1
        train_dataset.set_epoch(epoch)
        if accelerator.is_main_process:
            torch.save(accelerator.unwrap_model(model).state_dict(),os.path.join(config['output_dir'],f"epoch_{epoch}_step_{step}.pt"))


if __name__ == '__main__':
    main()
