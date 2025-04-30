import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def generate_and_save_keywords(input_file, output_file, yake_kw_extractor, top_n=7):
    df = pd.read_csv(input_file)
    
    def extract_keywords(text, top_n=top_n):
        keywords_with_scores = yake_kw_extractor.extract_keywords(text)
        # Extract only keywords (ignore scores)
        keywords = [kw for kw, _ in keywords_with_scores[:top_n]]
        return keywords

    # Apply extraction function and clean duplicates
    df['keywords'] = df['description'].astype(str).apply(
        lambda x: ' '.join(dict.fromkeys(' '.join(extract_keywords(x)).split()))
    )

    df.to_csv(output_file, index=False)
    print(f"Keywords extracted and saved to {output_file}")
    
    return df

def compute_and_save_embeddings(df, embedding_model, output_embedding_file):
    keyword_list = df['keywords'].tolist()
    
    # Generate embeddings
    keyword_embeddings = embedding_model.encode(keyword_list, convert_to_numpy=True)
    
    # Save embeddings
    np.save(output_embedding_file, keyword_embeddings)
    print(f"Embeddings saved to {output_embedding_file}")
    
    return keyword_embeddings

# Initialize embedding model
embedding_model = SentenceTransformer('paraphrase-mpnet-base-v2')

# Initialize YAKE! keyword extractor
yake_kw_extractor = yake.KeywordExtractor(lan="en", n=3)

# Extract keywords and save
df = generate_and_save_keywords('emojis.csv', 'emoji_data_with_keywords.csv', yake_kw_extractor)

# Compute embeddings and save
compute_and_save_embeddings(df, embedding_model, 'emoji_keyword_embeddings.npy')

