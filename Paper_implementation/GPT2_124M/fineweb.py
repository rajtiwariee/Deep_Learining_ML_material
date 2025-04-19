#download the fineweb-edu 10B dataset

import os
import multiprocessing as mp    
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
#--------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100 M tokens per shard


#create the cahche the local directory if it does not exist
DATA_CACHE_DIR = Path(local_dir)
DATA_CACHE_DIR.mkdir(parents = True, exist_ok = True)

#--------------------------------------Download the dataset--   
fw = load_dataset("HuggingFaceFW/fineweb-edu", name = remote_name, split = "train")

#init the toeknsizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] #end of text token
def tokenize(doc):
    #tokenize the document and returns a numpy array of uint16 tokens
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype = np.uint16)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Tokens must be in the range [0, 65535]"
    return tokens_np

def write_datafile(filename, tokens_np):
    with open(filename, "wb") as f:
        #write the tokens to the file
        f.write(tokens_np.tobytes())
        
# tokenize the dataset and write it to the local directory
nprocs = max(1,os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    
    shard_index = 0
    #preallocate buffer to hold current shard
    all_tokens = np.empty((shard_size,), dtype = np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        
        #is there enough space in the current shard for the new tokens
        if token_count + len(tokens) < shard_size:
            all_tokens[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            #update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, desc="Tokenizing", unit="tokens")
            progress_bar.update(len(tokens))
        else:
            #write the current shard and start a new one
            split = 'val' if shard_index == 0 else 'train'
            filename = DATA_CACHE_DIR / f"edufineweb_{split}_{shard_index:04d}.npy"
            #split the document into whatever fits in this shardl
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens)
            shard_index += 1
            if shard_index == 3: #we will only keep 4 shards #1 val and #3 train(for testing)
                break
            progress_bar = None
            #populate the next shard with the leftover of the current do
            all_tokens[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder
            
    #write any remaining tokens as the last shard
    if token_count != 0:
        split = 'val' if shard_index == 0 else 'train'
        filename = DATA_CACHE_DIR / f"edufineweb_{split}_{shard_index:04d}.npy"
        write_datafile(filename, all_tokens[:token_count])
        