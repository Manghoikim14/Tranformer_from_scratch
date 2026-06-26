import torch
import torch.nn as nn
from torch.utils.data import Dataset,Dataloader,random_split
from dataset import BilingualDataset,causal_mask
from model import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import Wordlevel
from tokenizers.trainers import WordlevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import pathlib
def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokernizer(config,ds,lang):
    tokenizer_path=PAth(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer=Tokenizer(Wordlevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer=Whitespace()
        trainer=WordlevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    else :
        tokenizer=Tokenizer.from_file(src(tokenizer_path))

    return tokenizer     

def get_ds(config):
    ds_raw=load_dataset('myset_book',f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')

    #Build tokenizer
    tokenizer_src=get_or_build_tokernizer(config,ds_raw,config['lang_src'])       
    tokenizer_tgt=get_or_build_tokernizer(config,ds_raw,config['lang_tgt']) 

    #keep 90% for training and 10% for validation
    train_ds_size=int(0.9 *len(ds_raw))
    val_ds_size==len(ds_raw)-train_ds_size
    train_ds_raw,val_ds_size=random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds=BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['seq_len'])
    val_ds=BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['seq_len'])

    max_len_src=0
    max_len_tgt=0

    for item in ds_raw:
        src_ids=tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids=tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src=max(max_len_src,len(src_ids))
        max_len_tgt=max(max_len_tgt,len(tgt_ids))

    print(f'Max length of source sentence:{max_len_src}') 
    print(f'Max length of target sentence:{max_len_tgt}') 

    train_dataloader=Dataloader(train_ds,batch_size=config['batch_size'],shuffle=True)
    val_dataloader=Dataloader(val_ds,batch_size=1,shuffle=True)

    return train_dataloader,val_dataloader,tokenizer_tgt


def get_model(config,vocab_src_len,vocab_tgt_len):
    model=build_transformer()

