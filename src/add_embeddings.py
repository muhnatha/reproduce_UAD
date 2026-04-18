"""
Add text embeddings to existing H5 files.
Loads region_matching from H5 files and generates embeddings using
any embedding type available in utils.vlm_utils.
"""
import os
import sys
import glob
import json
import ast

import h5py
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.vlm_utils import get_text_embedding_options
from utils.file_utils import store_or_update_dataset


def add_embeddings_to_h5(h5_path, embedding_type, text_embedding_func, verbose=True):
    """
    Add embeddings of the specified type to a single H5 file.
    
    Args:
        h5_path: Path to H5 file
        embedding_type: String key (e.g., 'embeddings_qwen3_vl')
        text_embedding_func: Function to encode text to embedding
        verbose: Print progress
    
    Returns:
        (num_instances_updated, num_instances_skipped)
    """
    updated = 0
    skipped = 0
    
    with h5py.File(h5_path, 'r+') as f:
        instance_keys = list(f.keys())
        
        for inst_key in instance_keys:
            instance = f[inst_key]
            
            if 'region_matching' not in instance:
                if verbose:
                    print(f"  Skipping {inst_key}: no region_matching")
                skipped += 1
                continue
            
            region_matching_json = instance['region_matching'][()]
            region_matching = json.loads(region_matching_json.decode('utf-8'))
            
            embedding_group_name = embedding_type
            
            if embedding_group_name in instance:
                del instance[embedding_group_name]
            embedding_group = instance.create_group(embedding_group_name)
            
            for descriptions_str, color in region_matching.items():
                description_list = ast.literal_eval(descriptions_str)
                embeddings = []
                
                for desc in description_list:
                    emb = text_embedding_func(desc)
                    if isinstance(emb, np.ndarray):
                        emb = emb.astype(np.float32)
                    else:
                        emb = np.array(emb, dtype=np.float32)
                    embeddings.append(emb)
                
                embedding_array = np.array(embeddings, dtype=np.float32)
                store_or_update_dataset(embedding_group, color, embedding_array)
            
            updated += 1
            if verbose and updated % 10 == 0:
                print(f"  Processed {updated} instances...")
        
        f.flush()
    
    return updated, skipped


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add text embeddings to existing H5 files"
    )
    parser.add_argument(
        '--embedding_type',
        type=str,
        required=True,
        choices=['embeddings_oai', 'embeddings_st', 'embeddings_bge_m3', 
                 'embeddings_mxbai', 'embeddings_qwen3_vl'],
        help='Type of embedding to generate'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='Directory containing h5/ folder (e.g., dataset/objaverse)'
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        default=None,
        help='Specific categories to process (default: all)'
    )
    
    args = parser.parse_args()
    
    h5_dir = os.path.join(args.base_dir, 'h5')
    if not os.path.exists(h5_dir):
        raise FileNotFoundError(f"h5 folder not found in {args.base_dir}")
    
    text_embedding_func = get_text_embedding_options(args.embedding_type)
    
    if args.categories:
        h5_paths = []
        for cat in args.categories:
            path = os.path.join(h5_dir, f'{cat}.h5')
            if os.path.exists(path):
                h5_paths.append(path)
            else:
                print(f"Warning: {path} not found, skipping")
    else:
        h5_paths = glob.glob(os.path.join(h5_dir, '*.h5'))
    
    print(f"Processing {len(h5_paths)} H5 files with embedding: {args.embedding_type}")
    print(f"Embedding function: {text_embedding_func.__name__}")
    print()
    
    total_updated = 0
    total_skipped = 0
    
    for h5_path in tqdm(h5_paths, desc="H5 files"):
        category = os.path.basename(h5_path).replace('.h5', '')
        updated, skipped = add_embeddings_to_h5(
            h5_path, 
            args.embedding_type, 
            text_embedding_func,
            verbose=False
        )
        total_updated += updated
        total_skipped += skipped
        tqdm.write(f"  {category}: {updated} updated, {skipped} skipped")
    
    print()
    print(f"Done! Total: {total_updated} instances updated, {total_skipped} skipped")


if __name__ == '__main__':
    main()
