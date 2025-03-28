import pandas as pd
from sentence_transformers import SentenceTransformer
from entity_recognition.entity_recognition_service import EntityRecognitionService
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json

class EmojiParser:
    def __init__(self):
        self.emoji_df = pd.read_csv("../data/emoji_data_with_entities.csv", encoding='utf-8')
        self.emoji_list = self.emoji_df["emoji"].values
        self.sbert_model = SentenceTransformer("paraphrase-mpnet-base-v2")
        self.entity_recognition_service = EntityRecognitionService()

    def save(self):
        self.emoji_df = pd.read_csv("emojis.csv")
        self.sbert_model = SentenceTransformer("paraphrase-mpnet-base-v2")
        self.entity_recognition_service = EntityRecognitionService()
        self.precompute_emoji_vectors()
        self.extract_emoji_entities()

    @property
    def emojis(self):
        return self.emoji_df["Emoji"].values

    def precompute_emoji_vectors(self) -> pd.DataFrame:
        """Compute SBERT embeddings for emoji descriptions and store in DataFrame."""

        # Concatenate name & description for embedding
        texts = [f"{row['name']} {row['description']}" for _, row in self.emoji_df.iterrows()]

        # Compute embeddings
        embeddings = self.sbert_model.encode(texts, convert_to_numpy=True)

        # Store embeddings properly in NumPy format
        self.emoji_df['vector'] = [vec for vec in embeddings]
        
        print("Emoji vectors precomputed and stored successfully.")

    def preprocess_text(self, text):
        """
        Preprocess the text before entity extraction.
        Customize this as needed (e.g., lowercasing, removing punctuation).
        """
        return text.lower().strip()

    def extract_emoji_entities(self):
        """Extract named entities from emoji descriptions, save entities, and compute entity embeddings."""
        unique_entities = set() 

        def process_row(row):
            emoji_text = self.preprocess_text(f"{row['name']} {row['description']}")
            entities = self.entity_recognition_service.extract_entities_from_models(emoji_text)
            return list(set(entities))  # Remove duplicates in current extraction

        # Use ThreadPoolExecutor to parallelize
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_row, row): idx for idx, (_, row) in enumerate(self.emoji_df.iterrows())}
            results = [None] * len(self.emoji_df)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting entities"):
                idx = futures[future]
                entities = future.result()
                results[idx] = entities

                # Add to the global unique_entities set
                unique_entities.update(entities)

        # Save entities per emoji in the DataFrame
        self.emoji_df["entities"] = results
        self.emoji_df.to_csv("emoji_data_with_entities.csv", index=False)
        print(f"Entities per emoji saved to emoji_data_with_entities.csv")

        print(f"Total unique entities extracted: {len(unique_entities)}")

        # Compute embeddings for unique entities
        unique_entities_list = sorted(unique_entities)
        entity_embeddings = self.sbert_model.encode(unique_entities_list, convert_to_numpy=True)

        # Create DataFrame for entity → embedding mapping
        entity_embeddings_df = pd.DataFrame({
            "entity": unique_entities_list,
            "embedding": [embedding.tolist() for embedding in entity_embeddings]
        })

        # Save to CSV
        entity_embeddings_df.to_csv("entity_embeddings.csv", index=False)
        print(f"Entity embeddings saved to entity_embeddings.csv")

