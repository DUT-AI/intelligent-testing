import os
import argparse
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from app.infrastructure.database.connection import SessionLocal
from app.infrastructure.database.models import Question

def parse_args():
    parser = argparse.ArgumentParser(description="Extract question embeddings and save directly to PostgreSQL database")
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
        help="Embedding dimension (supports MRL up to 1024)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Fetch questions from PostgreSQL database
    print("--- Connecting to database and fetching questions ---")
    session = SessionLocal()
    try:
        # Fetch id, content, and analysis of questions
        db_questions = session.query(Question.id, Question.content, Question.analysis).all()
        print(f"Successfully fetched {len(db_questions)} questions from database.")
    except Exception as e:
        print(f"Error querying database: {e}")
        return
    finally:
        session.close()

    if not db_questions:
        print("No questions found in the database. Exiting.")
        return
        
    # 2. Process questions into text list
    question_ids = []
    texts_to_embed = []
    
    print("--- Preparing text data for embedding ---")
    for q_id, content, analysis in db_questions:
        c_text = content if content else ""
        a_text = analysis if analysis else ""
        
        # Combine content and analysis to capture full question context
        combined_text = f"Đề bài: {c_text} [SEP] Lời giải: {a_text}"
        
        question_ids.append(q_id)
        texts_to_embed.append(combined_text)
        
    # 3. Load Qwen3 Embedding model
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    print(f"--- Loading model {model_name} ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    # SentenceTransformer handles the device and loading automatically
    model = SentenceTransformer(model_name, device=device)
    
    # 4. Generate embeddings
    print("--- Generating Qwen3 Embeddings ---")
    
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
    
    # 5. Update database directly
    print("--- Saving embeddings directly back to PostgreSQL ---")
    session = SessionLocal()
    try:
        # Prepare list of dicts for bulk update
        update_data = []
        for idx, q_id in enumerate(question_ids):
            # Convert embedding numpy array to a standard list of floats
            emb_list = [float(x) for x in embeddings[idx]]
            update_data.append({
                "id": q_id,
                "embedding": emb_list
            })
        
        # Perform bulk update
        print("Updating questions in database...")
        session.bulk_update_mappings(Question, update_data)
        session.commit()
        print(f"✅ Success! Updated {len(update_data)} questions with Qwen embeddings in PostgreSQL.")
    except Exception as e:
        session.rollback()
        print(f"Error saving to database: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    main()
