import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Any
from embedding.generate_emoji_via_embedding import load_models_and_data, find_emojis_for_each_keyword
from disambiguation.disambiguation_service import get_best_sense_by_embedding
from scale.generate_scale import EmojiScaler
import argparse
import os
import json

def generate_emoji_annotation_json(file_path: str, output_json_path: str, top_k: int = 5) -> str:
    df = pd.read_csv(file_path, header=None)
    table_description = df.iloc[0, 0]
    data = df.iloc[2:].reset_index(drop=True)
    data.columns = df.iloc[1]
    data.columns.name = None
    column_names = list(data.columns)

    numeric_cols = [col for col in column_names if pd.to_numeric(data[col], errors='coerce').notna().mean() > 0.8]
    categorical_cols = [col for col in column_names if col not in numeric_cols]

    column_contexts = {}
    for col in column_names:
        result = get_best_sense_by_embedding(table_description, col)
        if result == "Need more description please!":
            column_contexts[col] = col
        else:
            column_contexts[col] = col if isinstance(result, str) else f"{col} {result['definition']}"

    scaler = EmojiScaler()
    scale_results = {}

    def scale_task(col):
        try:
            return col, scaler.generate_best_llm_scale(column_contexts[col])
        except Exception as e:
            return col, {"error": str(e)}

    with ThreadPoolExecutor(max_workers=min(8, len(numeric_cols))) as executor:
        futures = {executor.submit(scale_task, col): col for col in numeric_cols}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scales"):
            col, result = future.result()
            scale_results[col] = result

    embedding_model, keyword_model, df_embed, desc_embeds, keyword_embeds = load_models_and_data()

    unique_val_queries = {
        (col, val): f"{val} in the column of: {column_contexts[col]} and table description: {table_description}"
        for col in categorical_cols
        for val in data[col].dropna().unique()
    }

    def keyword_task(context_text):
        result = find_emojis_for_each_keyword(
            query_text=context_text,
            df=df_embed,
            description_embeddings=desc_embeds,
            keyword_embeddings=keyword_embeds,
            embedding_model=embedding_model,
            keyword_model=keyword_model,
            desc_weight=0.2,
            keyword_weight=0.8,
            top_k=top_k
        )
        combined = defaultdict(float)
        for kw, lst in result.items():
            for item in lst:
                combined[item['emoji']] += item['similarity'] * (2 ** (len(kw.split()) - 1))
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

    value_to_emojis = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(keyword_task, ctx): key for key, ctx in unique_val_queries.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Categorical"):
            key = futures[future]
            try:
                value_to_emojis[key] = [e for e, _ in future.result()]
            except Exception as e:
                value_to_emojis[key] = {"error": str(e)}

    result_json = {
        "table_description": table_description,
        "column_emoji_scales": scale_results,
        "categorical_value_emojis": {
            col: {
                str(val): value_to_emojis.get((col, val), [])
                for val in data[col].dropna().unique()
            }
            for col in categorical_cols
        }
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    return result_json

def main():
    parser = argparse.ArgumentParser(description="Emoji Annotation Generator")

    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to input CSV file. Top-left cell is table description.')
    parser.add_argument('--output_json', type=str, required=True,
                        help='Path to output CSV with emoji annotations.')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top K emojis to show per cell.')

    args = parser.parse_args()

    # Run annotation pipeline
    generate_emoji_annotation_json(
        file_path=args.input_csv,
        output_json_path=args.output_json,
        top_k=args.top_k,
    )

if __name__ == '__main__':
    main()