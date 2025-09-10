import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import pickle

def write_datafile(filename, tokens_np):
    """Note: this write_datafile code is adapted from the data processing code of the modded-nanogpt 
    repository, which can be found at https://github.com/KellerJordan/modded-nanogpt/blob/master/data/fineweb.py """
    assert len(tokens_np) < 2**31
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20250429  # magic
    header[1] = 1          # version
    header[2] = len(tokens_np)
    tokens_np = np.asarray(tokens_np, dtype=np.uint16)
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_np.tobytes())

def create_shuffled_token_generator(dataset, permutation, encoder, eot_token):
    for doc_idx in permutation:
        text = dataset[int(doc_idx)].get("text", "").strip()
        if text:
            tokens = [eot_token] + encoder.encode_ordinary(text)
            yield np.array(tokens, dtype=np.uint16)

# Arg Parsing
parser = argparse.ArgumentParser(description="Data retrieval and document-based subsampling.")
parser.add_argument("-d", "--dataset", type=str, required=True)
parser.add_argument("-c", "--dataset_config", type=str, default=None)
parser.add_argument("--subsample_factor", type=float, default=1.0, help="Use 1/N of the dataset documents (e.g., 8 means 1/8th of documents).")
parser.add_argument("-s", "--shard_size", type=int, default=1_000_000_000, help="Size of each shard in tokens.")
parser.add_argument("-o", "--output_dir", type=str, required=True)
parser.add_argument("--shuffle_seed", type=int, default=42)
parser.add_argument("--dataset_split", type=str, default="train")
args = parser.parse_args()


os.makedirs(args.output_dir, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

# No streaming for loading in the data
print("Loading dataset...")
dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
num_docs = len(dataset)
print(f"Dataset loaded with {num_docs:,} documents.")

print(f"Creating global shuffle permutation with seed {args.shuffle_seed}...")
rng = np.random.default_rng(args.shuffle_seed)
full_permutation = np.arange(num_docs)
rng.shuffle(full_permutation)

# subsample using the number of documents
num_docs_to_process = int(num_docs / args.subsample_factor)
final_permutation = full_permutation[:num_docs_to_process]
print(f"Subsampling to process the first {len(final_permutation):,} documents from the shuffled set.")

print("\nProcessing and writing shards...")
token_generator = create_shuffled_token_generator(dataset, final_permutation, enc, eot)

shard_index = 0
total_tokens_written = 0
written_files = []

all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
token_count = 0
progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")

for tokens in token_generator:
    n_tokens_to_add = len(tokens)
    
    offset = 0
    while offset < n_tokens_to_add:
        space_in_shard = args.shard_size - token_count
        chunk_size = min(n_tokens_to_add - offset, space_in_shard)
        
        end_offset = offset + chunk_size
        all_tokens_np[token_count : token_count + chunk_size] = tokens[offset:end_offset]
        
        token_count += chunk_size
        progress_bar.update(chunk_size)
        offset = end_offset

        # if shard is full, write the shard and start a new one
        if token_count == args.shard_size:
            filename = os.path.join(args.output_dir, f"{args.dataset_split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np)
            written_files.append(filename)
            total_tokens_written += token_count
            shard_index += 1
            
            # reset
            token_count = 0
            progress_bar.close()
            progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")

# write the remaining tokens
progress_bar.close()
if token_count > 0:
    filename = os.path.join(args.output_dir, f"{args.dataset_split}_{shard_index:06d}.bin")
    write_datafile(filename, all_tokens_np[:token_count])
    written_files.append(filename)
    total_tokens_written += token_count

meta = {
    'vocab_size': enc.n_vocab,
    'encoder': 'gpt2',
    'dtype': 'uint16',
    'total_tokens': total_tokens_written,
    'shards': written_files,
    'subsample_factor': args.subsample_factor,
    'shuffle_seed': args.shuffle_seed,
}
meta_filename = os.path.join(args.output_dir, "meta.pkl")
with open(meta_filename, "wb") as f:
    pickle.dump(meta, f)

print("\n--- Preprocessing Complete ---")
print(f"Wrote {len(written_files)} shard(s) with a total of {total_tokens_written:,} tokens.")