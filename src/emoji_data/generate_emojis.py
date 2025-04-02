from collections import defaultdict
from disambiguation.disambiguation_service import get_best_sense_by_embedding
import argparse
import os
import pandas as pd
from embedding.generate_emoji_via_embedding import (
    load_models_and_data,
    find_emojis_for_each_keyword
)
from entity_recognition.emoji_ner_generation_service import EmojiNERGenerationService

def run_combined_service(column_texts: list, table_description: str, top_k: int = 10,
                         desc_weight: float = 0.4, keyword_weight: float = 0.6) -> dict:
    """
    Returns top emojis per column based on combined NER and embedding approaches.
    If a column is ambiguous, prompt the user for a description to proceed.
    """
    print(f"\nRunning Combined Emoji Service for {len(column_texts)} columns")
    full_contexts = [f"{col} in the context of {table_description}" for col in column_texts]
    print(f"Full contexts: {full_contexts}")

    # Load models and data
    embedding_model, keyword_model, df, desc_embeds, keyword_embeds = load_models_and_data()
    emoji_ner_service = EmojiNERGenerationService()

    # Step 1: NER
    print("\nStep 1: Running NER service...")
    ner_results_list = emoji_ner_service.get_emojis_for_text_list(full_contexts, top_n=top_k)

    # Step 2: Disambiguation + Filtering
    print("\nStep 2: Running disambiguation and preparing for embedding...")
    filtered_columns = []
    column_to_query_text = {}

    for col in column_texts:
        context = f"{col} {table_description}"
        result = get_best_sense_by_embedding(context, col)

        if result == "Need more description please!":
            print(f"Column \"{col}\" is ambiguous.")
            user_input = input(f"Please provide a short description for \"{col}\": ").strip()
            if user_input:
                column_to_query_text[col] = f"{col} {user_input}"
                filtered_columns.append(col)
            else:
                print(f"No description provided. Skipping \"{col}\".")
            continue

        if len(col.strip().split()) >= 2:
            column_to_query_text[col] = col
            filtered_columns.append(col)
        elif isinstance(result, dict) and result["score"] >= 0.1:
            column_to_query_text[col] = col + " " + result["definition"]
            filtered_columns.append(col)
        else:
            print(f"Column \"{col}\" is ambiguous.")
            user_input = input(f"Please provide a short description for \"{col}\": ").strip()
            if user_input:
                column_to_query_text[col] = f"{col} {user_input}"
                filtered_columns.append(col)
            else:
                print(f"No description provided. Skipping \"{col}\".")

    # Step 3: Embedding-based keyword matching
    print("\nStep 3: Running embedding-based keyword search...")
    keyword_results = {}
    for col in filtered_columns:
        query_text = column_to_query_text[col]
        print(query_text)
        matches = find_emojis_for_each_keyword(
            query_text=query_text,
            df=df,
            description_embeddings=desc_embeds,
            keyword_embeddings=keyword_embeds,
            embedding_model=embedding_model,
            keyword_model=keyword_model,
            desc_weight=desc_weight,
            keyword_weight=keyword_weight,
            top_k=top_k
        )
        keyword_results[col] = matches

    # Step 4: Combine results per column
    print("\nStep 4: Combining emoji scores per column...")
    combined_results = {}

    for i, col in enumerate(column_texts):
        combined_scores = defaultdict(float)

        # NER-based emojis
        ner_result = ner_results_list[i]
        for entity_text, emoji_list in ner_result.items():
            weight = 2 ** (len(entity_text.split()) - 1)
            for emoji_char, score in emoji_list:
                combined_scores[emoji_char] += score * weight

        # Keyword-based emojis
        if col in keyword_results:
            keyword_match_dict = keyword_results[col]
            print(keyword_match_dict)
            for keyword, matches in keyword_match_dict.items():
                weight = 2 ** (len(keyword.split()) - 1)
                for match in matches:
                    emoji_char = match['emoji']
                    combined_scores[emoji_char] += match['similarity'] * weight

        # Sort and pick top N
        sorted_emojis = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        combined_results[col] = dict(sorted_emojis[:max(top_k, top_k)])

    # Final display
    print("\n Final Emoji Results Per Column:")
    for col, emoji_scores in combined_results.items():
        print(f"\n Column: {col}")
        if not combined_results[col]:
            print("Need more description please!")
            continue
        for emoji, score in emoji_scores.items():
            print(f"  {emoji}: {score:.4f}")

    return combined_results


def load_column_names(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        return df.iloc[:, 0].dropna().astype(str).tolist()

    else:
        raise ValueError("Unsupported file format. Please use .txt or .csv.")



def main():
    parser = argparse.ArgumentParser(description="Combined Emoji Mapping CLI")
    parser.add_argument('--description', type=str, required=True,
                        help='The table-level description.')
    parser.add_argument('--column_file', type=str, required=True,
                        help='Path to a file containing column names (TXT or CSV).')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Top N emojis.')
    parser.add_argument('--desc_weight', type=float, default=0.4,
                        help='Weight for emoji description embeddings.')
    parser.add_argument('--keyword_weight', type=float, default=0.6,
                        help='Weight for emoji keyword embeddings.')

    args = parser.parse_args()

    print("Loading column names...")
    column_names = load_column_names(args.column_file)
    print(f"Loaded {len(column_names)} columns from {args.column_file}")

    print("\nRunning Combined Emoji Service...")
    results = run_combined_service(
        column_texts=column_names,
        table_description=args.description,
        top_k=args.top_k,
        desc_weight=args.desc_weight,
        keyword_weight=args.keyword_weight
    )

if __name__ == '__main__':
    main()


#print(run_combined_service(["bond", "love",], "Relationship"))
#print(run_combined_service(["bond", "exchanges"], "Financial Data"))
#print(run_combined_service(["bond", "exchanges"], ""))