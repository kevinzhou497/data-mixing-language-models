import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import pickle

def write_datafile(filename, toks):
    assert len(toks) < 2**31
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20250429  # magic
    header[1] = 1          # version
    header[2] = len(toks)
    toks = np.asarray(toks, dtype=np.uint16)
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())

# ----------------- ARG PARSING -------------------
parser = argparse.ArgumentParser(description="FineWeb-like dataset preprocessing")
parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset name (Hugging Face)")
parser.add_argument("-c", "--dataset_config", type=str, default=None, help="Optional dataset config")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Tokens per shard (ignored if --num_shards is used)")
parser.add_argument("-t", "--total_tokens", type=int, default=-1, help="Total number of tokens to generate (-1 for full dataset)")
parser.add_argument("--num_shards", type=int, default=None, help="Number of shards (overrides --shard_size)")
parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory for shards")
parser.add_argument("--no_val", action="store_true", help="Skip creating a validation shard")
parser.add_argument("--shuffle_seed", type=int, default=42, help="Seed for deterministic shuffling")
parser.add_argument("--dataset_split", type=str, default="train", required=True, help="Split for the loaded dataset")
args = parser.parse_args()

# ----------------- SETUP -------------------
os.makedirs(args.output_dir, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']
dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split, streaming=True)

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    return np.array(tokens, dtype=np.uint16)

def identity(x):
    return x

# ----------------- TOKENIZATION AND SHUFFLING -------------------
tokenized_docs = []
print("Tokenizing documents...")
for doc in dataset:
    text = doc.get("text", "").strip()
    if text:
        tokenized_docs.append(tokenize(doc))
    if args.total_tokens > 0 and sum(len(t) for t in tokenized_docs) >= args.total_tokens * 1.2:
        break

# Shuffle documents deterministically
rng = np.random.default_rng(args.shuffle_seed)
rng.shuffle(tokenized_docs)

# Clip total tokens if necessary
total_available_tokens = sum(len(t) for t in tokenized_docs)
if args.total_tokens == -1:
    args.total_tokens = total_available_tokens
else:
    args.total_tokens = min(args.total_tokens, total_available_tokens)

# Calculate shard size if using num_shards
if args.num_shards:
    args.shard_size = (args.total_tokens + args.num_shards - 1) // args.num_shards

# ----------------- SHARDING -------------------
nprocs = max(1, os.cpu_count() - 2)
written_files = []
total_tokens_written = 0
total_tokens_considered = 0

with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    stop = False

    for tokens in pool.imap_unordered(identity, tokenized_docs, chunksize=16):
        if stop:
            break

        tokens = tokens[:args.total_tokens - total_tokens_written]
        n_tokens = len(tokens)

        if token_count + n_tokens < args.shard_size:
            all_tokens_np[token_count:token_count+n_tokens] = tokens
            token_count += n_tokens
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(n_tokens)
            total_tokens_considered += n_tokens
        else:
            remainder = args.shard_size - token_count
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]

            split = "train" if (shard_index > 0 or args.no_val) else "val"
            filename = os.path.join(args.output_dir, f"{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np)
            written_files.append(filename)

            total_tokens_written += token_count + remainder
            total_tokens_considered += token_count + remainder
            progress_bar.update(remainder)
            shard_index += 1
            progress_bar = None

            spill = len(tokens) - remainder
            if spill > 0:
                all_tokens_np[0:spill] = tokens[remainder:]
                token_count = spill
            else:
                token_count = 0

        if total_tokens_written + token_count >= args.total_tokens:
            stop = True

    # leftover tokens
    remaining_tokens = min(token_count, args.total_tokens - total_tokens_written)
    if remaining_tokens > 0:
        split = "train" if (shard_index > 0 or args.no_val) else "val"
        filename = os.path.join(args.output_dir, f"{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:remaining_tokens])
        written_files.append(filename)
        total_tokens_written += remaining_tokens

# ----------------- SAVE META -------------------
meta = {
    'vocab_size': enc.n_vocab,
    'encoder': 'gpt2',
    'dtype': 'uint16',
    'total_tokens': total_tokens_written,
    'shards': written_files,
}
with open(os.path.join(args.output_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"Finished! Wrote {len(written_files)} shards, total tokens: {total_tokens_written:,}")
