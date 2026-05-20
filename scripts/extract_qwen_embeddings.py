import os
import json
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Extract question embeddings using Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument(
        "--input_json",
        type=str,
        default="data/raw/XES3G5M/XES3G5M/metadata/question_full.json",
        help="Path to the input JSON file containing questions"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/raw/XES3G5M/XES3G5M/metadata/qwen_embeddings_1024d.csv",
        help="Path to save the output embeddings CSV"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Inference batch size"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=1024,
        help="Embedding dimension to output (supports MRL up to 1024)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Qwen3 Embedding model
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    print(f"--- Loading model {model_name} ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    # SentenceTransformer handles the device and loading automatically
    model = SentenceTransformer(model_name, device=device)
    
    # 2. Read input JSON
    if not os.path.exists(args.input_json):
        raise FileNotFoundError(f"Input file not found at: {args.input_json}")
        
    print(f"Reading questions from: {args.input_json}")
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # 3. Process questions into text list
    question_ids = []
    texts_to_embed = []
    
    print("--- Preparing text data for embedding ---")
    for item_id, item_data in data.items():
        content = item_data.get("content", "")
        analysis = item_data.get("analysis", "")
        
        # Combine content and analysis to capture full question context
        # (matching the pattern used in the previous BERT/ModernBERT extraction)
        combined_text = f"Đề bài: {content} [SEP] Lời giải: {analysis}"
        
        question_ids.append(item_id)
        texts_to_embed.append(combined_text)
        
    print(f"Total questions to embed: {len(texts_to_embed)}")
    
    # 4. Generate embeddings
    print("--- Generating Qwen3 Embeddings ---")
    
    # Encode with sentence-transformers.
    # Note: Qwen3-Embedding-0.6B supports Matryoshka Representation Learning (MRL).
    # We can pass truncate_dim parameter if we want lower dimensions.
    encode_kwargs = {}
    if args.dimension != 1024:
        encode_kwargs["truncate_dim"] = args.dimension
        print(f"Truncating embeddings to dimension: {args.dimension}")
        
    embeddings = model.encode(
        texts_to_embed,
        batch_size=args.batch_size,
        show_progress_bar=True,
        **encode_kwargs
    )
    
    # 5. Save to CSV
    print(f"--- Saving embeddings to: {args.output_csv} ---")
    results = []
    for idx, item_id in enumerate(question_ids):
        row = {"ID": item_id}
        for dim_idx, val in enumerate(embeddings[idx]):
            row[f"Qwen_{dim_idx}"] = float(val)
        results.append(row)
        
    df = pd.DataFrame(results)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    
    print(f"✅ Success! Saved {len(df)} embeddings to: {args.output_csv}")

if __name__ == "__main__":
    main()
