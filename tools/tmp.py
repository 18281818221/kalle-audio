import yaml
from dy_dataset import TTSDataset_online_lance, get_lance_filelist
from transformers import AutoTokenizer
from train import split_file_lst

config = yaml.safe_load(open("./configs/vae_llama_offline-flow-mrte.yaml"))
tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
base_lst = get_lance_filelist(config["dataset"]["meta_path"])
base_lst = split_file_lst(base_lst,1,0)
train_dataset = TTSDataset_online_lance(config["dataset"],tokenizer,base_lst,'cuda',output_bf16=config['use_flash_attation'])
item = train_dataset[0]
import pdb;pdb.set_trace()
print('end.')

