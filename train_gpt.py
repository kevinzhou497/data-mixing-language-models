# File contents are originally from the https://github.com/KellerJordan/modded-nanogpt repository 
# Specifically, the train_gpt2.py file from the b345a1f commit ("new Muon record" ) on the master branch (October 11th, 2024)
# The file was then modified to fit the purposes of our optimal data mixing research work
import argparse
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

# ----------------------------------------------------------------------------------
# Arg Parsing 
parser = argparse.ArgumentParser(description="Training Modded NanoGPT Script")

parser.add_argument('--train_bin_primary', type=str, required=True, help='location of primary train.bin')
parser.add_argument('--train_bin_secondary', type=str, required=True, help='location of secondary train.bin')
parser.add_argument('--train_bin_third', type=str, default=None, help='location of third train.bin')
parser.add_argument('--mixing_ratio', type=float, default=None, help='ratio of primary dataset (e.g., 0.7 means 70% primary, 30% secondary)')
parser.add_argument('--mixing_ratios', type=float, nargs="+", default=None, help='proportions of the data sources when there are greater than two data sources')
parser.add_argument('--val_bin', type=str, required=True, help='location of desired val.bin')
parser.add_argument('--val_bin_2', type=str, default=None, help='location of the second desired val.bin')
parser.add_argument('--logdir', type=str, required=True, help='Directory to save logs and checkpoints')
parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate for training')
parser.add_argument('--seed', type=int, default=42, help='Seed')
parser.add_argument('--params', type=str, help='Number of parameters in the model (e.g. 124M)')
parser.add_argument('--num_iterations', type=int, default=42, help='Number of iterations')
parser.add_argument('--subsample', type=int, default=1, help='Subsample factor')
parser.add_argument('--hq_dataset', type=str, help='High-quality data being used (e.g. wikipedia)')
args = parser.parse_args()
# ----------------------------------------------------------------------------------
# For better performance and precision
torch.set_float32_matmul_precision('high')
# Setting seed
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# ----------------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    r"""
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1): # split grouped QKV parameters
                    g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    scale = g.size(1)**0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1))**0.5 # scale to have update.square().mean() == 1
                p.data.add_(g, alpha=-lr * scale)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model
class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / (2 * config.n_layer)**0.5)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model
@dataclass
class GPTConfig:
    vocab_size : int = 50257
    n_layer : int = 4 # changed from 12
    n_head : int = 12
    n_embd : int = 1024 # changed from 768 

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
     # only reads the header, returns header data
    with open(filename, "rb") as f:
         # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20250429:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        exit(1)
    # print(header)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens
 
def _load_data_shard(filename):
    with open(filename, "rb") as f:
         # first read the header, which is 256 int32 integers (4 bytes each)
         header = np.frombuffer(f.read(256*4), dtype=np.int32)
         assert header[0] == 20250429, "magic number mismatch in the data .bin file"
         assert header[1] == 1, "unsupported version"
         ntok = header[2] # number of tokens (claimed)
         # the rest of it are tokens, stored as uint16
         tokens = np.frombuffer(f.read(), dtype=np.uint16)

    #assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        # B = device batch size, T = sequence length
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            # assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files) # wraps around here if the shards have finished
        self.current_position = self.process_rank * self.B * self.T # starts the current position over
        self.tokens = _load_data_shard(self.files[self.current_shard])

    # B_override used if specific mixing ratio specified
    def next_batch(self, B_override=None):
        B = B_override if B_override is not None else self.B
        T = self.T
        needed = B * T + 1
        if self.current_position + needed > len(self.tokens):
            self.advance()

        buf = self.tokens[self.current_position : self.current_position + needed]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        if buf.numel() == 0:
            raise ValueError(f"Next batch would be empty: current_position={self.current_position}, needed={needed}, total_tokens={len(self.tokens)}")

        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes

        return x.cuda(), y.cuda()

class MixedDistributedDataLoader:
    def __init__(self, primary_loader, secondary_loader, total_batch_size, mix_ratio=None, third_loader=None, mixing_ratios=None):
        self.primary_loader = primary_loader
        self.secondary_loader = secondary_loader
        if third_loader:
            self.third_loader = third_loader
        else:
            self.third_loader = None
        self.mix_ratio = mix_ratio
        self.total_batch_size = total_batch_size
        self.mixing_ratios = mixing_ratios

    def reset(self):
        self.primary_loader.reset()
        self.secondary_loader.reset()
        if self.third_loader:
            self.third_loader.reset()

    def next_batch(self):
        # setting the ratio of data sources 1 and 2 in the batch
        # have to make it able to accept 3 data sources
        if self.mixing_ratios:
            B = self.total_batch_size 
            B1 = int(self.mixing_ratios[0]*B)
            B2 = int(self.mixing_ratios[1] * B)
            B3 = B - B1 - B2 
            x1, y1 = self.primary_loader.next_batch(B_override=B1)
            x2, y2 = self.secondary_loader.next_batch(B_override=B2)
            x3, y3 = self.third_loader.next_batch(B_override=B3)
            return torch.cat([x1, x2, x3], dim=0), torch.cat([y1, y2, y3], dim=0)
        else:
            B = self.total_batch_size
            B1 = int(self.mix_ratio * B)
            B2 = B - B1
            x1, y1 = self.primary_loader.next_batch(B_override=B1)
            x2, y2 = self.secondary_loader.next_batch(B_override=B2)
            return torch.cat([x1, x2], dim=0), torch.cat([y1, y2], dim=0)

    @property
    def ntok_total(self):
        if self.third_loader:
            return self.primary_loader.ntok_total + self.secondary_loader.ntok_total + self.third_loader.ntok_total
        else:  
            return self.primary_loader.ntok_total + self.secondary_loader.ntok_total

    @property
    def files(self):
        if self.third_loader:
            return self.primary_loader.files + self.secondary_loader.files + self.third_loader.files 
        else:     
            return self.primary_loader.files + self.secondary_loader.files

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    # add in a third data source
    train_bin_primary : str = args.train_bin_primary
    train_bin_secondary : str = args.train_bin_secondary
    train_bin_third : str = args.train_bin_third
    mixing_ratio : float = args.mixing_ratio
    mixing_ratios = args.mixing_ratios
    input_val_bin : str = args.val_bin # input .bin to eval validation loss on
    input_val_bin_2 : str = args.val_bin_2 # the second input .bin to eval validation loss on 
    # optimization hyperparams
    # total num training tokens = sequence_length * batch_size * num_iterations
    #batch_size : int = 8*64 # batch size, in sequences, across all devices
    batch_size = 128 # smaller to test
    device_batch_size : int = 128 # batch size, in sequences, per device
    sequence_length : int =  256 # sequence length, in tokens
    num_iterations : int = args.num_iterations # number of iterations to run
    subsample : int = args.subsample
    learning_rate : float = args.learning_rate
    warmup_iters : int = 0
    # warmdown: 1800 / 6200 * num_iterations ratio, since the original script had 1800 warmdown for 6200 total iterations
    warmdown_iters : int = args.num_iterations * 1800 / 6200  # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 100 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 131072 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
    params : str = args.params # how many parameters in the model, example: 124M, 1.5B, etc
    hq_dataset : str = args.hq_dataset 
args = Hyperparameters()

# set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available() # need cuda
dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(ddp_world_size)
print(f"using device: {device}")
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length

# load tokens
train_loader_primary = DistributedDataLoader(args.train_bin_primary, B, T, ddp_rank, ddp_world_size)
train_loader_secondary = DistributedDataLoader(args.train_bin_secondary, B, T, ddp_rank, ddp_world_size)
if args.train_bin_third:
    train_loader_third = DistributedDataLoader(args.train_bin_third, B, T, ddp_rank, ddp_world_size)
else:
    train_loader_third = None
    
if args.mixing_ratios:
    train_loader = MixedDistributedDataLoader(train_loader_primary, train_loader_secondary, third_loader=train_loader_third, mixing_ratios=args.mixing_ratios, total_batch_size=B)
else:
    train_loader = MixedDistributedDataLoader(train_loader_primary, train_loader_secondary, mix_ratio=args.mixing_ratio, total_batch_size=B)


# need a mixed loader for validation as well?
# make sure the data is all good here
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if args.input_val_bin_2:
    val_loader_2 = DistributedDataLoader(args.input_val_bin_2, B, T, ddp_rank, ddp_world_size)

# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
# changed to ensure that it doesn't crash if val set is too small
val_steps = min(args.val_tokens // (B * T * ddp_world_size), val_loader.ntok_total // (B * T * ddp_world_size))
if args.input_val_bin_2:
    val_steps_2 = min(args.val_tokens // (B * T * ddp_world_size), val_loader_2.ntok_total // (B * T * ddp_world_size))

# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
# overall batch size divided by (device batch size * world size)
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# init the model from scratch
num_vocab = 50257
# Specify the config here to change model size
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True # suggested by @Chillee
model = torch.compile(model)
# here we wrap model into DDP container
model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95)
optimizers = [optimizer1, optimizer2]
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# begin logging
if args.mixing_ratios:
    log_save_dir = f"wiki_pubmed/{args.num_iterations}iters/mix{args.mixing_ratios}_lr{args.learning_rate}_{args.params}_{args.subsample}"
else:
    log_save_dir = f"{args.hq_dataset}/{args.num_iterations}iters/mix{args.mixing_ratio}_lr{args.learning_rate}_{args.params}_{args.subsample}"
    
if master_process:
    #run_id = str(uuid.uuid4())
    run_id = str(log_save_dir)
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)    
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with torch.no_grad(): # of course, we'd like to use ctx here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        perplexity = torch.exp(val_loss)
        # log val loss to console, logfile, and summary file
        if master_process:
            print(f'val 1, step:{step}/{args.num_iterations} ppl:{perplexity:.4f} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'val 1, step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        
        # for the second val, if it is passed in
        if args.input_val_bin_2:
            val_loader_2.reset()
            val_loss_2 = 0.0
            for _ in range(val_steps_2):
                x_val, y_val = val_loader_2.next_batch()
                with torch.no_grad(): # of course, we'd like to use ctx here too, but that creates a torch.compile error for some reason
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss_2 += loss
            dist.all_reduce(val_loss_2, op=dist.ReduceOp.AVG)
            val_loss_2 /= val_steps_2
            perplexity_2 = torch.exp(val_loss_2)
            # log val loss to console, logfile, and summary file
            if master_process:
                print(f'val 2, step:{step}/{args.num_iterations} ppl:{perplexity_2:.4f} val_loss:{val_loss_2:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
                with open(logfile, "a") as f:
                    f.write(f'val 2, step:{step}/{args.num_iterations} val_loss:{val_loss_2:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
            
        
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        # Append run result to summary file, make this run specific so don't keep appending like this
        
        # Improve the summary log saving to differentiate all the experiments
        if args.mixing_ratios: # when we are using both Wikipedia and PubMed
            if master_process:
                summary_file = f"summary_logs/wiki_pubmed/summary_{args.num_iterations}iter_{args.params}_{args.subsample}.txt"
                with open(summary_file, "a") as f:
                    f.write(f"mixing_ratio={args.mixing_ratios}, learning_rate={args.learning_rate}, val_loss={val_loss:.4f}, perplexity={perplexity:.4f}, val_loss_2={val_loss_2:.4f}, perplexity_2={perplexity_2:.4f}\n")
            break
            
        else:
            if master_process:
                summary_file = f"summary_logs/{args.hq_dataset}/summary_{args.num_iterations}iter_{args.params}_{args.subsample}.txt"
                with open(summary_file, "a") as f:
                    f.write(f"mixing_ratio={args.mixing_ratio}, learning_rate={args.learning_rate}, val_loss={val_loss:.4f}, perplexity={perplexity:.4f}\n")
            break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with model.no_sync(): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
print(sum(p.numel() for p in model.parameters()))
print("end of script")
# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()
