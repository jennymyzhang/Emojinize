import argparse
from sentence_transformers import SentenceTransformer
from entity_recognition.entity_recognition_service import EntityRecognitionService
from emoji_data.emoji_information import EmojiParser
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import ast
import concurrent.futures


class EmojiNERGenerationService:
    def __init__(self):
        # Load emoji dataset and entity recognition service
        self.emoji_information = EmojiParser()
        self.entity_recognition_service = EntityRecognitionService()

        # Emoji dataframe from EmojiParser (make sure it aligns with description embeddings order)
        self.emoji_df = self.emoji_information.emoji_df

        # Load precomputed emoji description embeddings
        print("Loading emoji description embeddings...")
        self.emoji_description_embeddings = np.load('../data/emoji_description_embeddings.npy')

        # Load precomputed entity embeddings
        print("Loading entity embeddings from CSV...")
        self.entity_embedding_map = self.load_entity_embeddings('../data/entity_embeddings.csv')

        # Initialize SentenceTransformer for fallback encoding
        print("Loading SentenceTransformer for fallback embeddings...")
        self.sbert_model = SentenceTransformer("paraphrase-mpnet-base-v2")

        # Cache computed embeddings to avoid recomputation
        self.new_entity_cache: Dict[Tuple[str, str], np.ndarray] = {}

    def load_entity_embeddings(self, filepath: str) -> dict:
        df = pd.read_csv(filepath)
        entity_embed_map = {}

        for _, row in df.iterrows():
            # The CSV stores a tuple in string form, e.g. "('New York', 'LOC')"
            entity_tuple = ast.literal_eval(row['entity'])
            embedding = ast.literal_eval(row['embedding'])
            entity_embed_map[entity_tuple] = np.array(embedding, dtype=np.float32)

        print(f"Loaded {len(entity_embed_map)} entity embeddings.")
        return entity_embed_map

    def find_most_similar_embedding(self, query_vec: np.ndarray) -> Tuple[Tuple[str, str], np.ndarray, float]:
        query_norm = np.linalg.norm(query_vec) + 1e-9

        def compute_similarity(item):
            key, embedding = item
            embedding_norm = np.linalg.norm(embedding) + 1e-9
            sim = float(np.dot(query_vec, embedding) / (query_norm * embedding_norm))
            return key, embedding, sim

        best_match_key = None
        best_match_embedding = None
        best_similarity = -1.0

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(compute_similarity, item) for item in self.entity_embedding_map.items()]
            for future in concurrent.futures.as_completed(futures):
                key, embedding, sim = future.result()
                if sim > best_similarity:
                    best_similarity = sim
                    best_match_key = key
                    best_match_embedding = embedding

        return best_match_key, best_match_embedding, best_similarity

    def get_entity_embedding(self, entity_text: str, entity_label: str, similarity_threshold: float = 0.85) -> np.ndarray:
        entity_key = (entity_text, entity_label)

        # 1. Direct lookup
        if entity_key in self.entity_embedding_map:
            # Found precomputed embedding
            return self.entity_embedding_map[entity_key]

        # 2. Cache lookup
        if entity_key in self.new_entity_cache:
            return self.new_entity_cache[entity_key]

        # 3. Compute embedding on-the-fly
        combined_text = f"{entity_text} {entity_label}"
        computed_embedding = self.sbert_model.encode(combined_text)

        # 4. Compare to known embeddings
        best_match_key, best_match_embedding, best_similarity = self.find_most_similar_embedding(computed_embedding)

        if best_similarity >= similarity_threshold:
            final_embedding = best_match_embedding
        else:
            final_embedding = computed_embedding

        # 5. Cache and return
        self.new_entity_cache[entity_key] = final_embedding
        return final_embedding

    def get_top_emojis_for_entity(self, entity_text: str, entity_label: str, top_n: int = 3) -> List[Tuple[str, float]]:
        query_vec = self.get_entity_embedding(entity_text, entity_label)
        query_norm = np.linalg.norm(query_vec) + 1e-9

        similarities = []
        for idx, row in self.emoji_df.iterrows():
            emoji_char = row['emoji']
            emoji_vec = self.emoji_description_embeddings[idx]
            emoji_norm = np.linalg.norm(emoji_vec) + 1e-9

            # Cosine similarity with the emoji's textual description embedding
            description_sim = float(np.dot(query_vec, emoji_vec) / (query_norm * emoji_norm))

            # Optional: incorporate any known entities for the emoji
            emoji_entities = ast.literal_eval(row.get("entities", []))
            entity_sim = 0
            count = 0
            exact_match_boost = 0.0

            for emoji_entity, emoji_entity_label in emoji_entities:
                emoji_entity_vec = self.get_entity_embedding(emoji_entity, emoji_entity_label)
                emoji_entity_norm = np.linalg.norm(emoji_entity_vec) + 1e-9
                sim_score = float(np.dot(query_vec, emoji_entity_vec) / (query_norm * emoji_entity_norm))

                entity_sim += sim_score
                count += 1

                # Provide a small boost if there's an exact text match
                if entity_text.lower() == emoji_entity.lower():
                    exact_match_boost = 0.3

            entity_sim = (entity_sim / count) if count > 0 else 0

            # Weighted combination
            final_score = (0.6 * description_sim) + (0.4 * entity_sim) + exact_match_boost
            similarities.append((emoji_char, final_score))

        # Sort highest first
        similarities.sort(key=lambda x: x[1], reverse=True)
        # Return top-N
        return similarities[:top_n]

    def _extract_and_filter_entities(self, text: str) -> List[Tuple[str, str]]:
        raw_entities = list(self.entity_recognition_service.extract_named_entities(text))
        # Sort by length descending
        raw_entities.sort(key=lambda x: len(x[0]), reverse=True)

        filtered = []
        kept_strings = set()  # store lowercased forms we decide to keep

        for ent_text, ent_label in raw_entities:
            candidate_lower = ent_text.lower()
            # If this smaller text is already part of a bigger entity we kept, skip
            if any(candidate_lower in kept for kept in kept_strings):
                continue

            # Otherwise keep it
            filtered.append((ent_text, ent_label))
            kept_strings.add(candidate_lower)

        return filtered

    def get_emojis_for_text(self, text: str, top_n: int = 3):
        """
        Single-text version (unchanged except for substring filtering).
        Returns {entity_text: [(emoji, score), (emoji, score), ...]}
        """
        extracted_entities = self._extract_and_filter_entities(text)
        entity_emoji_candidates = []

        # We want up to `top_n` emojis per entity
        for entity_text, entity_label in extracted_entities:
            top_emojis = self.get_top_emojis_for_entity(entity_text, entity_label, top_n)
            entity_emoji_candidates.append((entity_text, top_emojis))

        # Build a dictionary {entity_text: [(emoji, score), ...]}
        result = {}
        for entity_text, emojis in entity_emoji_candidates:
            result[entity_text] = emojis
        return result

    def get_emojis_for_text_list(self, texts: List[str], top_n: int = 3) -> List[Dict[str, List[Tuple[str, float]]]]:
        # 1. Extract+filter entities in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(self._extract_and_filter_entities, txt): i
                for i, txt in enumerate(texts)
            }
            text_entities_list = [None] * len(texts)
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                text_entities_list[idx] = future.result()

        # 2. Compute frequency: (entity_text, entity_label) -> set of text indices
        appearance_map = {}
        for i, entity_list in enumerate(text_entities_list):
            for ent_text, ent_label in entity_list:
                key = (ent_text, ent_label)
                if key not in appearance_map:
                    appearance_map[key] = set()
                appearance_map[key].add(i)

        # We want to keep entities that appear in >= 30% of the texts
        threshold = 0.2
        min_count = int(np.ceil(threshold * len(texts)))
        valid_entities = {
            ent_key for ent_key, indices in appearance_map.items()
            if len(indices) >= min_count
        }

        # 3. For each text, filter out entities that don't meet the threshold
        #    Then we'll call get_top_emojis_for_entity for each retained entity.
        #    We'll do the emoji lookups in parallel as well.
        results = [{} for _ in range(len(texts))]

        def compute_emojis(ent_text, ent_label):
            return self.get_top_emojis_for_entity(ent_text, ent_label, top_n=top_n)

        for i, entity_list in enumerate(text_entities_list):
            # Filter only the "valid" ones
            retained = [
                (t, l) for (t, l) in entity_list
                if (t, l) in valid_entities
            ]
            # Now look up top emojis for these retained entities (concurrently)
            # We'll gather tasks, then populate results after completion.
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_map = {
                    executor.submit(compute_emojis, ent_text, ent_label): (ent_text, ent_label)
                    for (ent_text, ent_label) in retained
                }
                for future in concurrent.futures.as_completed(future_map):
                    (ent_text, ent_label) = future_map[future]
                    top_emojis = future.result()  # List[ (emoji, score), ... ]
                    results[i][ent_text] = top_emojis

        return results


def main():
    parser = argparse.ArgumentParser(description="Emoji NER Generation CLI")
    parser.add_argument('--texts', type=str, nargs='+', required=True,
                        help='Multiple texts to extract emojis for. Provide them separated by space or repeat --texts flag.')
    parser.add_argument('--top_n', type=int, default=3, help='Number of top emojis to return per entity.')

    args = parser.parse_args()

    texts = args.texts
    top_n = args.top_n

    print(f"Running EmojiNERGenerationService on {len(texts)} texts (Top N: {top_n})")

    emoji_service = EmojiNERGenerationService()

    # If you only have one text, you can still use get_emojis_for_text
    # But here we demonstrate the multiple-text approach:
    results_list = emoji_service.get_emojis_for_text_list(texts, top_n=top_n)

    for i, text in enumerate(texts):
        print(f"\n=== Results for Text {i+1}: \"{text}\" ===")
        for entity, emojis in results_list[i].items():
            print(f"  Entity: {entity}")
            for (emoji_char, score) in emojis:
                print(f"    Emoji: {emoji_char}, Score: {score:.4f}")


if __name__ == '__main__':
    main()
