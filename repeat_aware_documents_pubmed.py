import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset, get_dataset_split_names
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
        doc = dataset[int(doc_idx)]
        # retrieve the necessary PubMed abstract text
        article = doc.get("MedlineCitation", {}).get("Article", {})
        title = article.get("ArticleTitle", "").strip()

        abstract_obj = article.get("Abstract", {})
        abstract = abstract_obj.get("AbstractText", "").strip() if isinstance(abstract_obj, dict) else ""

        # only include if the abstract is present
        if abstract:
            text = f"{title}\n\n{abstract}" if title else abstract
            tokens = [eot_token] + encoder.encode_ordinary(text)
            yield np.array(tokens, dtype=np.uint16)

def process_and_write_split(split_name, dataset, indices, enc, eot, shard_size, output_dir):
    print(f"\nProcessing and writing shards for '{split_name}' split...")
    token_generator = create_shuffled_token_generator(dataset, indices, enc, eot)

    shard_index = 0
    total_tokens_written = 0
    written_files = []

    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {split_name}_{shard_index}")

    for tokens in token_generator:
        n_tokens_to_add = len(tokens)
        
        offset = 0
        while offset < n_tokens_to_add:
            space_in_shard = shard_size - token_count
            chunk_size = min(n_tokens_to_add - offset, space_in_shard)
            
            end_offset = offset + chunk_size
            all_tokens_np[token_count : token_count + chunk_size] = tokens[offset:end_offset]
            
            token_count += chunk_size
            progress_bar.update(chunk_size)
            offset = end_offset

            if token_count == shard_size:
                filename = os.path.join(output_dir, f"{split_name}_{shard_index:06d}.bin")
                write_datafile(filename, all_tokens_np)
                written_files.append(filename)
                total_tokens_written += token_count
                shard_index += 1
                
                token_count = 0
                progress_bar.close()
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {split_name}_{shard_index}")

    progress_bar.close()
    if token_count > 0:
        filename = os.path.join(output_dir, f"{split_name}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])
        written_files.append(filename)
        total_tokens_written += token_count
        
    print(f"Finished '{split_name}' split. Wrote {len(written_files)} shard(s) with {total_tokens_written:,} tokens.")
    return written_files, total_tokens_written

parser = argparse.ArgumentParser(description="Data retrieval and document-based subsampling for use with PubMed.")
parser.add_argument("-d", "--dataset", type=str, required=True)
parser.add_argument("-c", "--dataset_config", type=str, default=None)
parser.add_argument("--subsample_factor", type=float, default=1.0, help="Use 1/N of the dataset documents for training (e.g., 8 means 1/8th of documents).")
parser.add_argument("-s", "--shard_size", type=int, default=1_000_000_000, help="Size of each shard in tokens.")
parser.add_argument("-o", "--output_dir", type=str, required=True)
parser.add_argument("--shuffle_seed", type=int, default=42)
parser.add_argument("--dataset_split", type=str, default="train", help="The primary split to use (e.g., 'train').")
parser.add_argument("--val_split_percentage", type=float, default=0.01, help="Percentage of data to use for validation if no 'validation' split exists.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']
rng = np.random.default_rng(args.shuffle_seed)

# passing in this pull request for pubmed so that it retrieves data properly
revision_with_fix = "refs/pr/19"

print("Loading dataset info...")
available_splits = get_dataset_split_names(args.dataset, args.dataset_config, trust_remote_code=True, revision=revision_with_fix)
has_validation_split = "validation" in available_splits

train_indices, val_indices = None, None
train_dataset, val_dataset = None, None

if has_validation_split:
    print("Found existing 'train' and 'validation' splits.")
    train_dataset = load_dataset(args.dataset, args.dataset_config, split="train", trust_remote_code=True, revision=revision_with_fix)
    val_dataset = load_dataset(args.dataset, args.dataset_config, split="validation", trust_remote_code=True, revision=revision_with_fix)
    
    train_indices = np.arange(len(train_dataset))
    val_indices = np.arange(len(val_dataset))
    
    # shuffle
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    print(f"Train documents: {len(train_indices):,}, Validation documents: {len(val_indices):,}")

else:
    print(f"No 'validation' split found. Creating a split from '{args.dataset_split}' with {args.val_split_percentage:.2%} for validation.")
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split, trust_remote_code=True, revision=revision_with_fix)
    num_docs = len(dataset)
    
    full_permutation = np.arange(num_docs)
    rng.shuffle(full_permutation)
    
    # create train/val split from the shuffled indices
    split_point = int(num_docs * (1 - args.val_split_percentage))
    train_indices = full_permutation[:split_point]
    val_indices = full_permutation[split_point:]
    
    train_dataset = val_dataset = dataset
    print(f"Total documents: {num_docs:,}")
    print(f"Manually split into: Train ({len(train_indices):,}) and Validation ({len(val_indices):,})")

# subsample for the training data
num_train_docs_to_process = int(len(train_indices) / args.subsample_factor)
if num_train_docs_to_process < len(train_indices):
    train_indices = train_indices[:num_train_docs_to_process]
    print(f"Subsampling training data to {len(train_indices):,} documents.")

train_files, train_tokens = process_and_write_split(
    "train", train_dataset, train_indices, enc, eot, args.shard_size, args.output_dir
)

val_files, val_tokens = process_and_write_split(
    "val", val_dataset, val_indices, enc, eot, args.shard_size, args.output_dir
)
# meta data
meta = {
    'vocab_size': enc.n_vocab,
    'encoder': 'gpt2',
    'dtype': 'uint16',
    'shuffle_seed': args.shuffle_seed,
    'subsample_factor': args.subsample_factor,
    'train_shards': {
        'files': train_files,
        'total_tokens': train_tokens,
    },
    'val_shards': {
        'files': val_files,
        'total_tokens': val_tokens,
    }
}
meta_filename = os.path.join(args.output_dir, "meta.pkl")
with open(meta_filename, "wb") as f:
    pickle.dump(meta, f)

print("\n--- Preprocessing Complete ---")
print(f"Total training tokens: {train_tokens:,}")
print(f"Total validation tokens: {val_tokens:,}")