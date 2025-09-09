import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm
import tiktoken

def write_datafile(filename, tokens_np):
    """Note: this write_datafile code is taken from the data processing code of the modded-nanogpt 
    repository, which can be found at https://github.com/KellerJordan/modded-nanogpt/blob/master/data/fineweb.py """
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20250429; header[1] = 1; header[2] = len(tokens_np)
    tokens_np = np.asarray(tokens_np, dtype=np.uint16)
    with open(filename, "wb") as f:
        f.write(header.tobytes()); f.write(tokens_np.tobytes())

def iterate_documents(shard_files, eot_token):
    """Generator that yields one document at a time from a list of shard files."""
    for shard_file in shard_files:
        if not os.path.exists(shard_file) or os.path.getsize(shard_file) <= (256 * 4):
            continue
        tokens = np.fromfile(shard_file, dtype=np.uint16, offset=256 * 4)
        eot_indices = np.where(tokens == eot_token)[0]
        start_idx = 0
        for eot_idx in eot_indices:
            # The first token is the EOT token itself
            yield tokens[start_idx : eot_idx + 1]
            start_idx = eot_idx + 1

def write_docs_to_shards(docs, output_dir, shard_size, meta_info):
    """Writes a list of documents to a new set of sharded .bin files."""
    os.makedirs(output_dir, exist_ok=True)
    shard_index = 0
    tokens_in_current_shard = 0
    current_shard_tokens = np.empty((shard_size,), dtype=np.uint16)
    written_files = []
    total_tokens_written = 0

    print(f"Writing {len(docs):,} documents to {output_dir}...")
    # Note: later renamed the shard_*.bin files to either train_*.bin or validation_*.bin
    # * represents any number denoting the shard number, such as 000000, 000001, etc
    for doc_tokens in tqdm(docs, unit=" docs"):
        doc_len = len(doc_tokens)
        total_tokens_written += doc_len
        if tokens_in_current_shard + doc_len > shard_size:
            if tokens_in_current_shard > 0:
                filename = os.path.join(output_dir, f"shard_{shard_index:06d}.bin")
                write_datafile(filename, current_shard_tokens[:tokens_in_current_shard])
                written_files.append(filename)
                shard_index += 1
            tokens_in_current_shard = 0
        
        current_shard_tokens[tokens_in_current_shard : tokens_in_current_shard + doc_len] = doc_tokens
        tokens_in_current_shard += doc_len

    if tokens_in_current_shard > 0:
        filename = os.path.join(output_dir, f"shard_{shard_index:06d}.bin")
        write_datafile(filename, current_shard_tokens[:tokens_in_current_shard])
        written_files.append(filename)

    meta_info.update({
        'total_tokens': total_tokens_written,
        'shards': written_files,
    })
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta_info, f)
    print(f"Finished. Wrote {total_tokens_written:,} tokens to {len(written_files)} shard(s).")


def main():
    parser = argparse.ArgumentParser(description="Create nested, document-based subsamples from shards.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory of shards to subsample from.")
    parser.add_argument("-o", "--output_base_dir", type=str, required=True, help="Base directory to save the new subsample folders.")
    parser.add_argument("--fractions", type=float, nargs='+', required=True, help="List of fractions to generate (e.g., 0.5 0.25 0.125).")
    parser.add_argument("--shard_size", type=int, default=100_000_000, help="Size of each new shard in tokens.")
    args = parser.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc._special_tokens['<|endoftext|>']

    meta_path = os.path.join(args.input_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    
    original_shard_files = meta.get('shards', []) or meta.get('train_shards', {}).get('files', [])
    if not original_shard_files:
        print("Error: Could not find shard file list in meta.pkl")
        return

    print(f"Loading all documents from {args.input_dir} into memory to ensure nesting...")
    doc_iterator = iterate_documents(original_shard_files, eot_token)
    all_docs = list(tqdm(doc_iterator, desc="Loading docs"))
    total_docs = len(all_docs)
    print(f"Loaded a total of {total_docs:,} documents.")

    base_meta_info = {
        'vocab_size': meta['vocab_size'],
        'encoder': meta.get('encoder', 'gpt2'),
        'dtype': meta['dtype'],
    }

    for fraction in sorted(args.fractions, reverse=True):
        if not (0.0 < fraction <= 1.0):
            print(f"Skipping invalid fraction: {fraction}. Must be between 0 and 1.")
            continue

        num_docs_to_keep = int(total_docs * fraction)
        docs_subset = all_docs[:num_docs_to_keep]
        
        output_dir = os.path.join(args.output_base_dir, f"subsample_{fraction}")
        
        subset_meta = base_meta_info.copy()
        subset_meta['description'] = f"{fraction*100:.1f}% subsample ({num_docs_to_keep:,} docs) of {args.input_dir}"
        
        write_docs_to_shards(docs_subset, output_dir, args.shard_size, subset_meta)

    print("\n--- All subsamples created successfully. ---")

if __name__ == "__main__":
    main()