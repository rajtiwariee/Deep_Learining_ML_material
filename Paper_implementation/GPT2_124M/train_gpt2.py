from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
# --- Wandb Import ---
import wandb
import os
from dotenv import load_dotenv

load_dotenv()

#load env variables
os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY') #replace with your wandb api key
#view results
# https://wandb.ai/llm_dl/nanogpt-fineweb-example/runs/b8mi7j5m?nw=nwuserrajtiwariee

#--------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CausalSelfAttention(nn.Module):

    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        #key , query , value_projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.c_proj.NANOGPT_SCALE_INIT = 1
        #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        #torch.tril constructs a lower triangular part
        #which sets upper part to 0

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self,x):
        B, T,C = x.size()

        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim = 2)
        
        #(Batch_size, num_heads, Seq_len, Head_dimension)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        
        
        # #attention (materializes the large (T, T)) matrix for all the queries and keys
        # #do the dot matrix multiplication with key and query and multiply with the sqrt to make it stable
        # attention = torch.matmul(q, k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        # #here masked_fill tries (to replace the values where the boolean mask with the value)
        # #so if condition becomes True (means 0 is present and need to be masked) we replace it with (-inf)
        # attention = attention.masked_fill(self.bias[:,:,:T,:T] == 0 , float('-inf'))
        # attention = F.softmax(attention, dim = -1)
        # #(Batch_size, num_heads, seq, seq) x (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM) 
        # #(BATCH_SIZE, NUM_HEADS, SEQ, HEAD_DIM)
        # y = attention @ v 
        
        y = F.scaled_dot_product_attention(
            query = q,
            key = k,
            value = v,
            is_causal = True,
        ) # flash-attention implementation
        
        
        #re-assemble all head outputs side by side
        y = y.transpose(1,2).contiguous().view(B,T,C)

        #output projection
        y = self.c_proj(y)
        return y


#feed forward Linear layers
class MLP(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        #here we expand and then again reduce to make it learn more 
        # (Embed_size, expand_size) = (768, 4*768)
        self.c_fc = nn.Linear(config.n_embd,4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') #bit better then relu
        # (expand_size,Embed_size) = (4*768, 768)
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        #residual connection normalised_layer -> self.mlp + x
        x = x + self.mlp(self.ln_2(x)) 
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 #sequence length
    vocab_size: int = 50257  #Number of tokens 50,000 BPE merges + 256 bytes tokens + 1 <endoftext> token
    n_layer: int = 12 #Number of transformer blocks
    n_head: int = 4 #Number of attention heads
    n_embd: int = 768 #Embedding size
    
class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        
        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight #share the weights of the token embedding and the output layer
        
        #we use nn.Module function
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std = (2* self.config.n_layer) ** -0.5 #we are scaling on the number of layers
                #This will control the variance of the weights
                #its 2 times because of attention and mlp
                #with each layer we have sqrt(n) times of std added
                #if we wanna scale we divide it by 1/sqrt(n) this makes it 1
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)   
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.zeros_(module.weight) #not used in gpt2

    def forward(self,idx, targets = None):
        #idx = (batch_size, sequence_length)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        #forward the token and the position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) #shape (T,)
        pos_emb = self.transformer.wpe(pos) #positional embedding of shape   # (B, T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embedding of shape (T, n_embd) = (1024, 768)
        
        x = tok_emb + pos_emb # (B, T, n_embd)
         
        #forward the block to the transformer (attn and mlp)
        for block in self.transformer.h:
            x = block(x)
            
        #forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits , loss
    
    def configure_optimizers(self,weight_decay, learning_rate, device):
        #start with all of the candidate parameters
        # Apply weight decay to all parameters except biases and LayerNorm weights
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}
        #create optim groups Any parameters that is 2D will be weight decays 
        #i.e all weight tensors in matmuls + embeddings decay, all biases and layernorms do not decay
        decay_params = [p for n, p in param_dict.items() if len(p.shape) > 1]
        nodecay_params = [p for n, p in param_dict.items() if len(p.shape) == 1]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Num decayed parmeter tensors: {len(decay_params)}, with {num_decay_params} total elements")
        print(f"Num nodecayed parmeter tensors: {len(nodecay_params)}, with {num_nodecay_params} total elements")
        
        #create AdamW optimizer and use the fused version if available
        #fused helps to fuse all the kernels into single one kernel
        #this will help to reduce the number of read/writes to the GPU and the operation will be faster
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        print(f'Using fused AdamW: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps = 1e-8,fused=use_fused)
        return optimizer
    
    @classmethod
    def from_pretrained(cls, model_type, override_args = None):
        """
        Load a pre-trained model weights from huggingface
        """
        assert model_type in ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" %model_type)
        
        #n_layer, n_head and n_embd are determined by the model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),#124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),#350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),#774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),#1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 #always 50257 for gpt model checkpoints
        config_args['block_size'] = 1024
        # config_args['bias'] = True #bias is always true for gpt model checkpoints
        
        # if 'dropout' in override_args:
        #     print(f"Overriding dropout to {override_args['dropout']}")
        #     config_args['dropout'] = override_args['dropout']
        #create a from_scaratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] #discard this mask/ buffer
        
        #init a huggingface / transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        #copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()  
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] #ignore the mask
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] #ignore the mask
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys) , f" mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the conv1d weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
                    
            else:
                #vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
             
        return model
    


#----------------------------- DataLoader Lite --------------------------------
import tiktoken
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

def load_tokens(filename):
    npt = np.fromfile(filename, dtype=np.uint16) #loads the raw binary
    ptt = torch.tensor((npt.astype(dtype=np.int32)), dtype=torch.long)
    return ptt

class DataLoaderLite(Dataset):
    
    def __init__(self, B, T , process_rank, num_processes,split):
        super().__init__()
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}, "split must be train or val"
        
        #load the data
        # with open('input.txt', 'r') as f:
        #     text = f.read()
        # #encode the text
        # enc = tiktoken.get_encoding("gpt2")
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # print(f"Loaded {len(self.tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens)// (B*T)} batches")
        
        #get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [os.path.join(data_root, shard) for shard in shards if shard.endswith('.npy')]
        shards.sort()
        self.shards = shards
        assert len(shards) > 0, "No shards found in the data directory"
        if master_process:
            print(f"Found {len(shards)} shards in the data directory for split {split}")
            
        #state init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        #state
        self.current_position = self.B * self.T * self.process_rank
        
    def __len__(self):
        return len(self.tokens) // (self.B * self.T)
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = buf[:-1].view(B,T) # (4, 256) = 4*256 = 1024
        y = buf[1:].view(B,T) #(4,32)
        self.current_position += B*T * self.num_processes
        
        #if loading the next batch goes out of the range of the tokens reset
        if self.current_position + (B*T*self.num_processes +1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

    def __getitem__(self, index):
        #get the next batch
        x, y = self.next_batch()
        return x, y


#----------------------Training on shakespeare dataset-------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#-----------------------------------Distributed DataParallelism--------------------------------
#simple launch
#python train.py
#DDP launch
#torchrun --standalone --nproc_per_node=8 train.py

#run the training loop 
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
#setup the ddp (distributed data parallel)
#torchrun command set the env vairables RANK, LOCAL_RANK, WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 #is this a ddp run?
if ddp:
    #use of DDP atm demands CUDA, we set the device appropriately according to the rank
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend= 'nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 #this process will be logging,checkpointing 
else:
    #vanilla, non-ddp run
    ddp_rank = 0
    ddp_local_rank = 0  
    ddp_world_size = 1
    master_process = True
    #attempt to autodetect the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    

import torch
import time
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    
    
total_batch_size = 524288 # 2^19 , ~0.5M in tokens
B = 4
T = 1024
assert total_batch_size % (B*T*ddp_world_size) == 0, f"total_batch_size {total_batch_size} is not divisible by B*T {B*T*ddp_world_size}"
grad_accumulation_steps = total_batch_size // (B*T*ddp_world_size) #this is the number of steps we need to accumulate the gradients
if master_process:
    print(f'Total desired batch size : {total_batch_size}')
    print(f'Total grad accumulation steps : {grad_accumulation_steps}')
    



train_loader = DataLoaderLite(B = B, T = T, process_rank= ddp_rank, num_processes = ddp_world_size,split = 'train')


#this sets it from float32 to TF32 -> in which instead of 32 we have 19 bits 
#making more TFLOps increasing 8 times from float32
torch.set_float32_matmul_precision('high') #set the precision to high for matmul
# print('set tfloat32')
#get the logits
model = GPT(GPTConfig(vocab_size=50304)) #we increased the vocab size to 50304 to make it nice number and divisible by powers of 2
model.train() #set the model to training model
model.to(device) #move the model to GPU
if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])
    #this will sync all the gradients that are accumulated across all the ranks
raw_model = model.module if ddp else model


#1. It reads your code in one go and compiles it to a single graph
#2. It optimizes the graph for performance and improves the speed and read/writes on the gpu
#3. Read/writes basically so if we perform a simple operation like having a gelu formula( 0.5 * math.pow(2, 0.5) * (x + math.pow(x, 3)))
#4. so if its normal interpreter it will first copy math.pow(2, 0.5) to the gpu and then perform the operation and then copy it back
#5. But with torch.compile it will do all the operations in one go and then copy it back to the gpu
#6. So it will reduce the number of read/writes to the gpu and increase the speed
model = torch.compile(model) #faster training of the model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 2 # 10% of max_steps
max_steps = 10 # 19073

log_interval = 10 #every 10 seconds
weight_decay = 1e-1


if master_process:
    wandb.init(
        project="nanogpt-fineweb-example", # Change project name as desired
        config={
            "total_batch_size": total_batch_size,
            "micro_batch_size_per_gpu": B,
            "sequence_length": T,
            "grad_accumulation_steps": grad_accumulation_steps,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "weight_decay": weight_decay,
            "ddp_world_size": ddp_world_size,
            "device": device,
        }
    )
    print(f'Total desired batch size: {total_batch_size}')
    print(f'Micro batch size (B*T): {B*T}')
    print(f'Gradient accumulation steps: {grad_accumulation_steps}')
    print(f"Max steps: {max_steps}")

    print(f"Max learning rate: {max_lr}")
    print(f"Min learning rate: {min_lr}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Weight decay: {weight_decay}")
    print(f"Device: {device}")
    print(f"DDP world size: {ddp_world_size}")



def get_lr(it):
    #1. Linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1)/ warmup_steps
    #2. if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    #3. In between use cosine decay down to min learning rate
    decay_ratio = (it- warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) #coeff start at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr) #this will give us the learning rate

#define the optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas = (0.9, 0.95), weight_decay=1e-8)
optimizer = model.configure_optimizers(weight_decay = 0.1 , learning_rate = 6e-4, device = device)
for step in range(50):
    t0 = time.time()
    optimizer.zero_grad() #reset the gradients
    loss_accum = 0.0
    for micro_step in range(grad_accumulation_steps): 
        x,y = train_loader.next_batch() #get the next batch
        x,y = x.to(device), y.to(device) #move the batch to GPU
        #forward the model
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
            #forward the model
        logits, loss = model(x,y)
        loss = loss / grad_accumulation_steps
        loss_accum += loss.detach()
        if ddp:
            model.required_backward_grad_sync = (micro_step == grad_accumulation_steps - 1)
        #send loss backward
        loss.backward() #calculate the gradients
        #this prevents the model from getting bigger shock of gradients (which means higher loss it could be because of the any unlucky batch of the data)
    if ddp:
        dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #clip the gradients
    #determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step() #update the weights
    if device == 'cuda':
        torch.cuda.synchronize() #synchronize the GPU
    t1 = time.time()
    dt = (t1-t0) * 1000 #convert to ms
    tokens_per_second = train_loader.B * train_loader.T *grad_accumulation_steps * ddp_world_size/ (dt/1000)
    # --- Wandb Logging (Master Process Only) ---
    if master_process and (step % log_interval == 0 or step == max_steps - 1):
        log_data = {
            "train/loss": loss_accum.item(),
            "train/norm": norm.item(),
            "train/lr": lr,
            "perf/dt_ms": dt,
            "perf/tokens_per_sec": tokens_per_second,
            "step": step,
        }
        wandb.log(log_data)
        print(f"Step {step:5d} | Loss: {loss_accum.item():.4f} | Norm: {norm.item():.4f} | LR: {lr:.4e} | dt: {dt:.2f}ms | Tok/sec: {tokens_per_second:,.0f}")


if master_process:
    wandb.finish() #finish the wandb run               

if ddp:
    destroy_process_group() #destroy the process group


import sys; sys.exit(0) #exit the program

#-----------------------------------------------Inference------------------------------------------------
# num_return_sequences = 5
# max_length = 30

# model = GPT.from_pretrained('gpt2') #this will load the weights of the model
# model.eval() #set the model to evaluation mode
# model.to(device) #move the model to GPU


# #prefix tokens
# import tiktoken
# enc = tiktoken.get_encoding("gpt2")
# tokens = enc.encode("Hello , I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long) #(8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8) -> repeat (repeats the tokens in 0 th dimension)
# x = tokens.to(device) #move the tokens to GPU

# #generate! right now x is (B, T) = (5, 8)
# #set the seed to 42
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     #forward the model
#     with torch.inference_mode():
#         logits = model(x) #(B, T, vocab_size)
#         #take the logits at the last position
#         logits = logits[:, -1, :] #(B, vocab_size) -> (5,1,50257)
#         #get the probabilities 
#         probs = F.softmax(logits, dim=-1)
#         #do top-k sampling of 50 (huggingface pipeline default)
#         #topk_probs here becomes (5, 50), topk_indices becomes (5, 50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         #select a token from the top-k probabilities
#         ix = torch.multinomial(topk_probs, num_samples=1)
#         #gather the corresponding indices
#         xcol = torch.gather(topk_indices, dim=-1, index=ix) #(5, 1)
#         #append to the sequence
#         x = torch.cat((x, xcol), dim=1)
        
# #print the generated text
# for i in range(num_return_sequences):
#     #decode the tokens to text
#     tokens = x[i, :max_length].tolist()
#     text = enc.decode(tokens)
#     print(f"> {text}")
    