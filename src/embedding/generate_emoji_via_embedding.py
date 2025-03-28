import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

def load_models_and_data():
    print(" Loading models and data...")

    # Initialize models
    embedding_model = SentenceTransformer('paraphrase-mpnet-base-v2')
    keyword_model = KeyBERT(model=embedding_model)

    # Load data
    df = pd.read_csv('../data/emoji_data_with_sentiment.csv')
    description_embeddings = np.load('../data/emoji_description_embeddings.npy')
    keyword_embeddings = np.load('../data/emoji_keyword_embeddings.npy')

    return embedding_model, keyword_model, df, description_embeddings, keyword_embeddings

def find_emojis_for_each_keyword(
    query_text,
    df,
    description_embeddings,
    keyword_embeddings,
    embedding_model,
    keyword_model,
    desc_weight=0.4,
    keyword_weight=0.6,
    top_k=10
):
    # Step 1: Extract keywords from query_text
    query_keywords = keyword_model.extract_keywords(
        query_text,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=2
    )
    
    # Display extracted keywords
    extracted_keywords = [kw for kw, _ in query_keywords]

    # Step 2: Prepare combined dataset embeddings
    combined_dataset_embeddings = (
        desc_weight * description_embeddings + keyword_weight * keyword_embeddings
    )

    all_results = {}

    # Step 3: For each keyword, find top_k emojis
    for keyword in extracted_keywords:

        # Embed the keyword
        keyword_embedding = embedding_model.encode(keyword, convert_to_numpy=True)

        # Compute similarity with the dataset embeddings
        similarities = cosine_similarity(
            keyword_embedding.reshape(1, -1),
            combined_dataset_embeddings
        )

        # Get top_k indices
        top_indices = np.argsort(similarities[0])[::-1][:top_k]

        # Collect results for this keyword
        results = []
        for idx in top_indices:
            row = df.iloc[idx]
            result = {
                'emoji': row['emoji'],
                'description': row['description'],
                'keywords': row['keywords'],
                'similarity': similarities[0][idx]
            }
            results.append(result)

        # Save results under the keyword
        all_results[keyword] = results

    return all_results

def get_emojis_for_texts(
    query_text,
    embedding_model,
    keyword_model,
    df,
    description_embeddings,
    keyword_embeddings,
    top_k=10,
    desc_weight=0.4,
    keyword_weight=0.6
):
    matches = find_emojis_for_each_keyword(
        query_text=query_text,
        df=df,
        description_embeddings=description_embeddings,
            keyword_embeddings=keyword_embeddings,
            embedding_model=embedding_model,
            keyword_model=keyword_model,
            desc_weight=desc_weight,
            keyword_weight=keyword_weight,
            top_k=top_k
    )
    return matches



def main():
    parser = argparse.ArgumentParser(description="Emoji Finder CLI Tool")
    parser.add_argument('--query', type=str, required=True, help='Query text to search emojis for.')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top emojis per keyword.')
    parser.add_argument('--desc_weight', type=float, default=0.4, help='Weight for description embeddings.')
    parser.add_argument('--keyword_weight', type=float, default=0.6, help='Weight for keyword embeddings.')

    args = parser.parse_args()

    query_text = args.query
    top_k = args.top_k
    desc_weight = args.desc_weight
    keyword_weight = args.keyword_weight

    print(f"\nRunning Emoji Search")
    print(f"Query: {query_text}")
    print(f"Top K: {top_k} | Desc Weight: {desc_weight} | Keyword Weight: {keyword_weight}")

    # Load everything
    embedding_model, keyword_model, df, description_embeddings, keyword_embeddings = load_models_and_data()

    # Run the search
    keyword_emoji_matches = find_emojis_for_each_keyword(
        query_text=query_text,
        df=df,
        description_embeddings=description_embeddings,
        keyword_embeddings=keyword_embeddings,
        embedding_model=embedding_model,
        keyword_model=keyword_model,
        desc_weight=desc_weight,
        keyword_weight=keyword_weight,
        top_k=top_k
    )

    # Display results
    print("\nResults")
    for keyword, matches in keyword_emoji_matches.items():
        print(f"\nKeyword: {keyword}")
        print(f"Top {len(matches)} matching emojis:")
        for match in matches:
            print(f"  {match['emoji']} (Similarity: {match['similarity']:.4f})")

if __name__ == '__main__':
    main()
