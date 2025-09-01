
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('numba').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

from itertools import cycle
import yaml
import os
import time
import torch.multiprocessing
import random

from copy import deepcopy
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, get_scheduler, AutoTokenizer
from torch.optim import AdamW

from dy_dataset import TTSDataset_online_lance_for_offlineFlow,DynamicBatchGenerator,dict_to_device, get_lance_filelist
from model import Llasa_onlineflow
from tqdm.auto import tqdm
from data_pool import DataPrefetchPool, DynamicPrefetchBatchIterator

# torch.multiprocessing.set_start_method('spawn', force=True)

def split_file_lst(file_lst,world_size,rank):
    average = len(file_lst)//world_size
    return file_lst[rank*average:(rank+1)*average]


def main():
    config = yaml.safe_load(open("./configs/vae_llama_online-flow.yaml"))
    config['exp_dir'] = os.path.join(config['exp_dir'],config['project_name'])
    config['log_dir'] = os.path.join(config['exp_dir'],'logs')
    config['output_dir'] = os.path.join(config['exp_dir'],'output')
    config['resume_dir'] = os.path.join(config['exp_dir'],'resume')

    os.makedirs(config['exp_dir'],exist_ok=True)
    os.makedirs(config['log_dir'],exist_ok=True)
    os.makedirs(config['output_dir'],exist_ok=True)
    os.makedirs(config['resume_dir'],exist_ok=True)

    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=config["log_dir"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"]
    )
    accelerator.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
    accelerator.deepspeed_plugin.deepspeed_config['gradient_accumulation_steps'] = config["gradient_accumulation_steps"]
    accelerator.init_trackers(project_name=config["project_name"])

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    
    base_lst = get_lance_filelist(config["dataset"]["meta_path"])
    print("Shuffling file list ......")
    random.seed(42)
    random.shuffle(base_lst)
    print(f"Shuffling done {base_lst[0]}")
    base_lst = split_file_lst(base_lst,accelerator.num_processes,accelerator.process_index)
    train_dataset = TTSDataset_online_lance_for_offlineFlow(config["dataset"],tokenizer,base_lst,accelerator.device,output_bf16=config['use_flash_attation'])

    model = Llasa_onlineflow(config['model'],tokenizer,flow=deepcopy(train_dataset.generator.flow),use_flash_attention=config['use_flash_attation'])
    import pdb;pdb.set_trace()
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

    last_checkpoint = None
    if config.get("start_checkpoint") is not None :
        assert os.path.exists(config.get("start_checkpoint"))
        last_checkpoint = config.get("start_checkpoint")
        print(f"-------------------------Start training from {last_checkpoint}")
    elif os.path.isdir(config['output_dir']):
        checkpoints = [os.path.join(config['output_dir'], d) for d in os.listdir(config['output_dir']) if d.startswith("epoch_")]
        if len(checkpoints) > 0:
            # Get the most recent checkpoint based on modification time
            last_checkpoint = max(checkpoints, key=os.path.getmtime)
            epoch = int(last_checkpoint.split('/')[-1].split('_')[1])
    
    if last_checkpoint is not None:
        print(f"-------------------------Loading model and tokenizer from checkpoint {last_checkpoint}")
        model.load_state_dict(torch.load(last_checkpoint,map_location='cpu'),strict=True)

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

    while(True):

        for i in tqdm(range(0, len(train_dataset), 1), disable=not accelerator.is_local_main_process): 
            item = data_pool.data_queue.get()
            data_pool.data_queue.task_done()

            batch = batch_generator.batch_add_item(item)
            if batch is None:
                continue

            batch_record_lst.append(len(batch))
            batch = dict_to_device(train_dataset.collate(batch), accelerator.device)

            with accelerator.accumulate(model):

                optimizer.zero_grad()
                outputs = model(
                    input_ids=              batch.get("input_ids"),
                    audio_latents=          batch.get("audio_latents"),
                    audio_distribution_l=   batch.get("distribute_lables"),

                    ids_mask=               batch.get("text_ids_mask"),
                    audio_mask=             batch.get("audio_latents_mask"),
                    target_mask=            batch.get("distribute_lables_mask"),
                    end_mask=               batch.get("enddist_mask"),
                )
                audio_loss = outputs['audio_loss']
                end_loss = outputs['end_loss']

                total_loss = audio_loss*config["audio_loss_weight"] + end_loss*config["end_loss_weight"]
 
                accelerator.backward(total_loss)
                optimizer.step()
                scheduler.step()

            accelerator.log({"total_loss": total_loss.detach().item()} ,step = step)
            accelerator.log({"audio_loss": audio_loss.detach().item()} ,step = step)
            accelerator.log({"end_loss": end_loss.detach().item()} ,step = step)

            if step != 0 and step % config["save_interval"] == 0:
                if accelerator.is_main_process:
                    torch.save(accelerator.unwrap_model(model).state_dict(),os.path.join(config['output_dir'],f"epoch_{epoch}_step_{step}.pt"))
                # accelerator.save_state(f"{config['resume_dir']}/step_{step}")

            if step % 50 == 0:
                accelerator.print("{}: Epoch:{}, Step:{}, batch_size:{}, lr:{}, total_loss:{}, audio_loss:{}, end_loss:{}，data_queue_size:{}".format(
                    time.ctime(),
                    epoch,
                    step,
                    sum(batch_record_lst) / len(batch_record_lst),
                    optimizer.param_groups[0]['lr'],
                    total_loss.detach().item(),
                    audio_loss.detach().item(),
                    end_loss.detach().item(),
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
