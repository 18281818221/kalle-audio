
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
import random
import torch.distributions as D

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, get_scheduler, AutoTokenizer
from torch.optim import AdamW

from dy_dataset import TTSDataset_online_lance,DynamicBatchGenerator,dict_to_device, get_lance_filelist
from model import Llasa,Llasa_random_drop_spkcond
from tqdm.auto import tqdm
from data_pool import DataPrefetchPool, DynamicPrefetchBatchIterator

# torch.multiprocessing.set_start_method('spawn', force=True)

def split_file_lst(file_lst,world_size,rank):
    average = len(file_lst)//world_size
    return file_lst[rank*average:(rank+1)*average]


def main():
    config_file = "./configs/vae_llama_offline-flow-ecapatdnn_v2.yaml"
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

    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=config["log_dir"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"]
    )
    accelerator.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
    accelerator.deepspeed_plugin.deepspeed_config['gradient_accumulation_steps'] = config["gradient_accumulation_steps"]
    accelerator.init_trackers(project_name=config["project_name"])

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    model = Llasa_random_drop_spkcond(config['model'],tokenizer,use_flash_attention=config['use_flash_attation'])

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
    base_lst = get_lance_filelist(config["dataset"]["meta_path"])
    print("Shuffling file list ......")
    random.seed(42)
    random.shuffle(base_lst)
    print(f"Shuffling done {base_lst[0]}")
    
    base_lst = split_file_lst(base_lst,accelerator.num_processes,accelerator.process_index)
    train_dataset = TTSDataset_online_lance(config["dataset"],tokenizer,base_lst,accelerator.device,output_bf16=config['use_flash_attation'])
    reader,i = train_dataset.info_lines[0]
    datas = reader.get_datas_by_rows([i])
    print(datas[0].text)

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

    print('accelerator prepare.....')
    model, optimizer,  scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )

    batch_record_lst = []
    batch_generator = DynamicBatchGenerator(hparams=config["batch_generator"])

    data_pool = DataPrefetchPool(
        dataset=train_dataset,
        max_size=config['datapool']['max_size'],
        num_workers=config['datapool']['num_workers'],
    )

    data_pool.start()

    # prefetch_size = config['datapool'].get('prefetch_size',0)
    # prefetch_size = min(prefetch_size,config['datapool']['max_size']*0.85)
    # progress_bar = tqdm(total=prefetch_size, desc="Filling queue", unit="item")
    # while data_pool.data_queue.qsize() < prefetch_size:
    #     current_size = data_pool.data_queue.qsize()
    #     progress_bar.n = current_size  # Update the progress bar with the current size
    #     progress_bar.refresh()  # Refresh the progress bar display
    #     # time.sleep(0.1)

    # print(f"Process {accelerator.process_index} prefetch {prefetch_size} done")
    # accelerator.wait_for_everyone()

    while(True):

        for i in tqdm(range(0, len(train_dataset), 1), disable=not accelerator.is_local_main_process): 
            item = data_pool.data_queue.get()
            data_pool.data_queue.task_done()

            batch = batch_generator.batch_add_item(item)
            if batch is None:
                continue

            batch_record_lst.append(len(batch))
            batch = dict_to_device(train_dataset.collate(batch), accelerator.device)
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

                    mels=                   batch.get("mels"),
                    speaker_cond_keep=      batch.get("speaker_cond_keep"),
                )
                audio_loss = outputs['audio_loss']
                end_loss = outputs['end_loss']

                mean = outputs['pre_mean']
                logs_scale = outputs['pre_log_scale']
                target_mask = batch.get("distribute_lables_mask")
                latent = torch.rand_like(mean) * torch.exp(logs_scale) + mean
                latent = latent.transpose(1,2)
                mask = target_mask.unsqueeze(1)

                mean_label, logs_scale_label = batch.get("distribute_lables").chunk(2, dim=-1)
                latent_label = torch.rand_like(mean_label) * torch.exp(logs_scale_label) + mean_label
                latent_label = latent_label.transpose(1,2)

                with torch.no_grad():
                    with torch.autocast(device_type="cuda"):
                        z_p = train_dataset.generator.flow(latent,mask)
                        z = train_dataset.generator.flow(latent_label,mask)
                # import pdb;pdb.set_trace()
                z_p = z_p.transpose(1,2)
                z = z.transpose(1,2)
                label_n = D.Normal(z, torch.exp(logs_scale_label))
                flow_n = D.Normal(z_p, torch.exp(logs_scale))

                kl = D.kl_divergence(flow_n, label_n)
                kl = kl.sum(2) / batch.get("audio_latents").shape[-1]
                kl = (kl * target_mask).sum() / target_mask.sum()

                total_loss = (audio_loss*config["audio_loss_weight"] 
                              + end_loss*config["end_loss_weight"]
                              + kl*config["kl_loss_weight"])
 
                accelerator.backward(total_loss)
                optimizer.step()
                scheduler.step()

            accelerator.log({"total_loss": total_loss.detach().item()} ,step = step)
            accelerator.log({"audio_loss": audio_loss.detach().item()} ,step = step)
            accelerator.log({"end_loss": end_loss.detach().item()} ,step = step)
            accelerator.log({"kl_loss": kl.detach().item()} ,step = step)

            if step != 0 and step % config["save_interval"] == 0:
                if accelerator.is_main_process:
                    torch.save(accelerator.unwrap_model(model).state_dict(),os.path.join(config['output_dir'],f"epoch_{epoch}_step_{step}.pt"))
                # accelerator.save_state(f"{config['resume_dir']}/step_{step}")

            if step % 50 == 0:
                accelerator.print("{}: Epoch:{}, Step:{}, batch_size:{}, lr:{}, total_loss:{}, audio_loss:{}, end_loss:{}，kl_loss:{}, data_queue_size:{}".format(
                    time.ctime(),
                    epoch,
                    step,
                    sum(batch_record_lst) / len(batch_record_lst),
                    optimizer.param_groups[0]['lr'],
                    total_loss.detach().item(),
                    audio_loss.detach().item(),
                    end_loss.detach().item(),
                    kl.detach().item(),
                    data_pool.data_queue.qsize()
                    ) )
                batch_record_lst = []

            step += 1
        
        epoch += 1
        train_dataset.set_epoch(epoch)
        if accelerator.is_main_process:
            torch.save(accelerator.unwrap_model(model).state_dict(),os.path.join(config['output_dir'],f"epoch_{epoch}_step_{step}.pt"))


if __name__ == '__main__':
    main()
