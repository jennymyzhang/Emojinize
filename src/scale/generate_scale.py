import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
from nltk.corpus import wordnet

class EmojiScaler:
    def __init__(self,antonym_pairs=None):
        self.emoji_df = pd.read_csv("../data/emoji_data_with_descriptions.csv")
        embedding_path = "../data/emoji_description_embeddings.npy"
        self.model = SentenceTransformer("paraphrase-mpnet-base-v2")

        # Use MPS device if available
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Load precomputed embeddings to MPS/CPU
        description_embeddings = np.load(embedding_path)
        tensor_embeddings = torch.tensor(description_embeddings)
        self.emoji_df["embedding"] = [x for x in tensor_embeddings.cpu().tolist()]

        if antonym_pairs is None:
            self.antonym_pairs = [
                ("empty", "full"), ("low", "high"), ("sad", "happy"), ("cold", "hot"),
                ("weak", "strong"), ("sick", "healthy"), ("dim", "bright"), ("off", "on"),
                ("start", "finish"), ("slow", "fast"), ("unsafe", "safe"), ("poor", "rich"),
                ("bad", "good"), ("dark", "light"), ("dull", "sharp"), ("quiet", "loud"),
                ("shallow", "deep"), ("short", "long"), ("dry", "wet"), ("flat", "juiced"),
                ("exhausted", "energized"),("inactive", "active"),
                ("boring", "exciting"), ("confusing", "clear"), ("dirty", "clean"),
                ("closed", "open"), ("declining", "growing"), ("unstable", "stable"),
                ("cold", "warm"), ("soft", "hard"), ("tight", "loose"), ("hard", "easy"),
                ("sad", "joyful"), ("small", "large"), ("tiny", "huge"), ("early", "late"),
                ("difficult", "simple"), ("good", "bad"), ("weak", "powerful"),
            ]
        else:
            self.antonym_pairs = antonym_pairs

    def _select_best_axis(self, column_name):

        def get_antonym(word):
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    for ant in lemma.antonyms():
                        return ant.name()
            return None

        # Try WordNet-based antonym for words in column name
        antonym = get_antonym(column_name)
        if antonym:
            return (antonym, column_name)

        # Fallback to predefined pairs
        query_vec = self.model.encode(column_name, convert_to_tensor=True).to(self.device)
        best_pair = None
        best_score = -float("inf")

        for a, b in self.antonym_pairs:
            a_vec = self.model.encode(a, convert_to_tensor=True).to(self.device)
            b_vec = self.model.encode(b, convert_to_tensor=True).to(self.device)
            midpoint = (a_vec + b_vec) / 2.0
            axis_vec = b_vec - a_vec

            midpoint_score = torch.cosine_similarity(query_vec, midpoint.unsqueeze(0), dim=1).item()
            direction_score = torch.cosine_similarity(query_vec, axis_vec.unsqueeze(0), dim=1).item()
            combined_score = 0.5 * midpoint_score + 0.5 * direction_score

            if combined_score > best_score:
                best_score = combined_score
                best_pair = (a, b)

        print(best_pair)
        return best_pair

    def generate_scale(self, column_name: str, top_k=20, scale_size=5):
        query_vec = self.model.encode(column_name, convert_to_tensor=True).to(self.device)

        # Step 1: Select best antonym axis
        antonym_pair = self._select_best_axis(column_name)
        print(f"Selected axis for '{column_name}': {antonym_pair[0]} → {antonym_pair[1]}")

        a_vec = self.model.encode(antonym_pair[0], convert_to_tensor=True).to(self.device)
        b_vec = self.model.encode(antonym_pair[1], convert_to_tensor=True).to(self.device)
        axis_vec = b_vec - a_vec
        axis_unit = axis_vec / (axis_vec.norm() + 1e-6)
        midpoint = (a_vec + b_vec) / 2.0

        # Step 2: Filter relevant emoji
        all_embeddings = torch.stack([torch.tensor(x).to(self.device) for x in self.emoji_df["embedding"]])
        similarities = util.pytorch_cos_sim(query_vec, all_embeddings)[0]
        top_indices = similarities.topk(top_k).indices.tolist()
        selected = self.emoji_df.iloc[top_indices].copy()
        selected_embeddings = torch.stack([torch.tensor(x).to(self.device) for x in selected["embedding"]])

        # Step 3: Projection and relevance
        centered = selected_embeddings - midpoint
        projection = torch.matmul(centered, axis_unit)
        relevance = util.pytorch_cos_sim(query_vec, selected_embeddings)[0]

        # Normalize both components
        norm_proj = (projection - projection.mean()) / (projection.std() + 1e-6)
        norm_rel = (relevance - relevance.mean()) / (relevance.std() + 1e-6)

        # Final score: projection dominates, relevance helps fine-tune
        selected["score"] = (0.85 * norm_proj + 0.15 * norm_rel).tolist()

        # Step 4: Sort and pick evenly spaced
        sorted_df = selected.sort_values(by="score", ascending=True).reset_index(drop=True)
        indices = torch.linspace(0, len(sorted_df) - 1, steps=scale_size).long()
        return sorted_df.iloc[indices]["emoji"].tolist()


    def apply_scale_to_column(self, df, col_name, scale_size=5):
        emoji_scale = self.generate_scale(col_name, scale_size=scale_size)
        col = df[col_name]
        min_val, max_val = col.min(), col.max()

        def tag(value):
            norm = (value - min_val) / (max_val - min_val + 1e-6)
            idx = int(norm * (len(emoji_scale) - 1))
            return emoji_scale[idx]

        return df[col_name].apply(tag)

    
scaler = EmojiScaler()
print(scaler.generate_scale("mood", scale_size=5))
print(scaler.generate_scale("battery", scale_size=5))
print(scaler.generate_scale("popularity", scale_size=5))