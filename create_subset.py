import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm
import tiktoken

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

def iterate_documents(shard_files, eot_token):
    """
    Generator that yields one document (as a numpy array) at a time from a list of shard files.
    Handles documents that may span across shard boundaries.
    """
    leftover = np.array([], dtype=np.uint16)
    for shard_file in shard_files:
        # check if file exists
        if not os.path.exists(shard_file) or os.path.getsize(shard_file) <= (256 * 4):
            continue
        
        # Read the tokens from the shard, skipping the header
        tokens = np.fromfile(shard_file, dtype=np.uint16, offset=256 * 4)
        
        # Concatenate the leftover tokens from the previous shard with the current tokens
        tokens = np.concatenate((leftover, tokens))
        
        eot_indices = np.where(tokens == eot_token)[0]

        # If no eot indices, the whole shard is part of a single document
        if len(eot_indices) == 0:
            leftover = tokens
            continue

        # Yield each complete document
        start_idx = 0
        for eot_idx in eot_indices:
            # Document is from the previous EOT (or start) to this EOT
            doc = tokens[start_idx:eot_idx]
            if start_idx > 0 or leftover.size == 0:
                yield np.concatenate(([eot_token], doc))
            start_idx = eot_idx + 1 
        
        leftover = tokens[start_idx:]


def main():
    parser = argparse.ArgumentParser(description="Create a document-based subset from existing shards.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory containing the original shards and meta.pkl file.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save the new subset shards.")
    parser.add_argument("--target_tokens", type=int, default=100_000_000, help="Approximate number of tokens for the subset.")
    parser.add_argument("--shard_size", type=int, default=200_000_000, help="Size of each new shard in tokens.")
    parser.add_argument("--split", type=str, help="train or val")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc._special_tokens['<|endoftext|>']

    # Loading in the existing meta data
    meta_path = os.path.join(args.input_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        print(f"Error: meta.pkl not found in {args.input_dir}")
        return

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    
    original_shard_files = meta[f'{args.split}_shards']['files']
    print(f"Found {len(original_shard_files)} original shards.")

    # Get the documents until the target token number is reached
    print(f"Collecting documents to reach approximately {args.target_tokens:,} tokens...")
    selected_docs_tokens = []
    total_tokens_collected = 0
    doc_iterator = iterate_documents(original_shard_files, eot_token)

    for doc_tokens in tqdm(doc_iterator, unit=" docs"):
        selected_docs_tokens.append(doc_tokens)
        total_tokens_collected += len(doc_tokens)
        if total_tokens_collected >= args.target_tokens:
            break
    
    print(f"Collected {len(selected_docs_tokens):,} documents with a total of {total_tokens_collected:,} tokens.")
    print("Writing new subset shards...")
    shard_index = 0
    tokens_in_current_shard = 0
    current_shard_tokens = np.empty((args.shard_size,), dtype=np.uint16)
    written_files = []

    for doc_tokens in tqdm(selected_docs_tokens, unit=" docs"):
        doc_len = len(doc_tokens)
        # If the doc doesn't fit, write the current shard and start a new one
        if tokens_in_current_shard + doc_len > args.shard_size:
            if tokens_in_current_shard > 0:
                filename = os.path.join(args.output_dir, f"shard_{shard_index:06d}.bin")
                write_datafile(filename, current_shard_tokens[:tokens_in_current_shard])
                written_files.append(filename)
                shard_index += 1
            tokens_in_current_shard = 0
        
        # Add the document to the current shard
        current_shard_tokens[tokens_in_current_shard : tokens_in_current_shard + doc_len] = doc_tokens
        tokens_in_current_shard += doc_len

    # Write the final partial shard
    if tokens_in_current_shard > 0:
        filename = os.path.join(args.output_dir, f"shard_{shard_index:06d}.bin")
        write_datafile(filename, current_shard_tokens[:tokens_in_current_shard])
        written_files.append(filename)

    # New meta data for this subset
    new_meta = {
        'description': f'Subset of approximately {args.target_tokens:,} tokens from {args.input_dir}',
        'vocab_size': enc.n_vocab,
        'encoder': 'gpt2',
        'dtype': 'uint16',
        'total_tokens': total_tokens_collected,
        'shards': written_files,
    }
    new_meta_filename = os.path.join(args.output_dir, "meta.pkl")
    with open(new_meta_filename, "wb") as f:
        pickle.dump(new_meta, f)

    print("\n--- Subsetting Complete ---")
    print(f"Wrote {len(written_files)} new shard(s) to {args.output_dir}")
    print(f"Total tokens in subset: {total_tokens_collected:,}")

if __name__ == "__main__":
    main()