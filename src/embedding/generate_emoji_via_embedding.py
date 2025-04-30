import argparse
from ast import Tuple
import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT


OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openai/gpt-4o-mini"

def call_openrouter(prompt: str, model: str = LLM_MODEL) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }

    response = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=15)

    if response.status_code != 200:
        raise ValueError(f"LLM call failed: {response.status_code} - {response.text}")

    response_json = response.json()
    return response_json["choices"][0]["message"]["content"].strip()

def generate_llm_description(value: str, column: str, table_description: str) -> str:
    prompt = f"""
    You are generating an expressive, emoji-matchable description for the value "{value}", 
    which appears in the column "{column}" within the table titled "{table_description}".

    Your task is to:
    - Identify any **real-world associations** of the value (e.g., BWI → Washington DC, Amazon → e-commerce).
    - Describe these associations using **concrete, literal, and visually grounded** terms.
    - Emphasize **objects, places, or symbols** commonly represented by emojis (e.g., planes, stars, trucks, boxes, tools, money).
    - Avoid vague emotional or symbolic language like "excited travelers" or "memorable moments".

    Good descriptions should:
    - Mention related cities, regions, or organizations if relevant.
    - Highlight things that can be visually seen, touched, or felt — not ideas.

    **Examples**:
    - For "Amazon" in column "Place": "Amazon is a company known for delivery boxes, warehouses, and online shopping"
    - For "Score" in column "Rating": "Scores are shown using stars, hearts, and thumbs up"
    - For "BWI" in column "Airport": "BWI is an airport near Washington DC, with planes, runways, and control towers"

    Return one single sentence. No bullet points, no commentary — just the description.
    """
    return call_openrouter(prompt)

def extract_triplet_from_query(query_text: str) -> Tuple:
    # Expects: "{value} in the context of {column_context} and table {table_description}"
    parts = query_text.split(" in the column of: ")
    value = parts[0].strip()
    col_and_table = parts[1].split(" and table ")
    column = col_and_table[0].strip()
    table_description = col_and_table[1].strip()
    return value, column, table_description

def load_models_and_data():
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
    # Step 1: Use OpenRouter to enhance the query
    value, column, table_description = extract_triplet_from_query(query_text)
    llm_description = generate_llm_description(value, column, table_description)

    # Step 2: Embed LLM description
    desc_embedding = embedding_model.encode(llm_description, convert_to_numpy=True)

    # Step 3: Extract keywords from the LLM-enhanced description
    query_keywords = keyword_model.extract_keywords(
        llm_description,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=3
    )
    extracted_keywords = [kw for kw, _ in query_keywords]

    # Step 4: Compute similarities
    # a) Description similarity (1 vector vs. emoji description matrix)
    desc_sim = cosine_similarity(
        desc_embedding.reshape(1, -1),
        description_embeddings
    )[0]  # shape: (n_emojis,)

    # b) Keyword similarity (average over all extracted keyword embeddings)
    if extracted_keywords:
        keyword_embeds = np.array([
            embedding_model.encode(kw, convert_to_numpy=True)
            for kw in extracted_keywords
        ])
        keyword_mean = keyword_embeds.mean(axis=0)
        keyword_sim = cosine_similarity(
            keyword_mean.reshape(1, -1),
            keyword_embeddings
        )[0]  # shape: (n_emojis,)
    else:
        keyword_sim = np.zeros_like(desc_sim)

    # Step 5: Combine similarities
    combined_sim = desc_weight * desc_sim + keyword_weight * keyword_sim

    # Step 6: Rank top_k
    top_indices = np.argsort(combined_sim)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            'emoji': row['emoji'],
            'description': row['description'],
            'keywords': row['keywords'],
            'similarity': combined_sim[idx]
        })

    return {llm_description: results}

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
