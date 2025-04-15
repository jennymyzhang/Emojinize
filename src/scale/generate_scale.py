import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import requests
import re
from typing import List, Tuple

OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "google/gemini-2.0-flash-001"

class EmojiScaler:
    def __init__(self, emoji_csv_path, embedding_path):
        self.emoji_df = pd.read_csv(emoji_csv_path)
        self.model = SentenceTransformer("paraphrase-mpnet-base-v2")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        description_embeddings = np.load(embedding_path)
        self.emoji_df["embedding"] = [x for x in torch.tensor(description_embeddings).cpu()]

    def generate_scale_with_axis(self, column_name: str, axis_pair: Tuple[str, str], top_k=100, scale_size=5) -> List[str]:
        # Normalize the antonym vectors before computing axis
        # and create an expanded projection-relevance scoring scheme with a balance of semantic similarity, directionality, and contextual emphasis
        # Normalize the antonym vectors before computing axis
        query_vec = self.model.encode(column_name, convert_to_tensor=True).to(self.device)
        a_vec = self.model.encode(axis_pair[0], convert_to_tensor=True).to(self.device)
        b_vec = self.model.encode(axis_pair[1], convert_to_tensor=True).to(self.device)

        a_vec = a_vec / (a_vec.norm() + 1e-6)
        b_vec = b_vec / (b_vec.norm() + 1e-6)

        query_vec = self.model.encode(column_name, convert_to_tensor=True).to(self.device)
        a_vec = self.model.encode(axis_pair[0], convert_to_tensor=True).to(self.device)
        b_vec = self.model.encode(axis_pair[1], convert_to_tensor=True).to(self.device)

        axis_vec = b_vec - a_vec
        axis_unit = axis_vec / (axis_vec.norm() + 1e-6)
        axis_unit = axis_vec / (axis_vec.norm() + 1e-6)
        midpoint = (a_vec + b_vec) / 2.0

        all_embeddings = torch.stack([torch.nn.functional.normalize(torch.tensor(x).to(self.device), dim=0) for x in self.emoji_df["embedding"]])
        similarities = util.pytorch_cos_sim(query_vec, all_embeddings)[0]
        top_indices = similarities.topk(top_k).indices.tolist()
        selected = self.emoji_df.iloc[top_indices].copy()
        selected_embeddings = torch.stack([torch.nn.functional.normalize(torch.tensor(x).to(self.device), dim=0) for x in selected["embedding"]])

        centered = selected_embeddings - midpoint
        projection = torch.matmul(centered, axis_unit)
        relevance = util.pytorch_cos_sim(query_vec, selected_embeddings)[0]

        norm_proj = (projection - projection.mean()) / (projection.std() + 1e-6)
        norm_rel = (relevance - relevance.mean()) / (relevance.std() + 1e-6)

        semantic_boost = torch.cosine_similarity(selected_embeddings, b_vec.unsqueeze(0), dim=1)
        norm_semantic_boost = (semantic_boost - semantic_boost.mean()) / (semantic_boost.std() + 1e-6)

        final_score = (0.5 * norm_proj + 0.3 * norm_rel + 0.2 * norm_semantic_boost)
        selected["emoji_text"] = selected["emoji"] + " (" + np.round(final_score.cpu().numpy(), 2).astype(str) + ")"
        selected["score"] = final_score.tolist()
        sorted_df = selected.sort_values(by="score", ascending=True).reset_index(drop=True)
        indices = torch.linspace(0, len(sorted_df) - 1, steps=scale_size).long()
        return sorted_df.iloc[indices]["emoji_text"].tolist()

    def _call_openrouter(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        body = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5
        }
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=15)

        if response.status_code != 200:
            raise ValueError(f"LLM call failed: {response.status_code} - {response.text}")

        response_json = response.json()
        if "choices" not in response_json:
            raise ValueError(f"Unexpected response format: {response_json}")

        return response_json["choices"][0]["message"]["content"].strip()

    def get_llm_antonym_pairs(self, column_name: str) -> List[Tuple[str, str]]:
        prompt = f"""
        Generate 2 antonym phrase pairs that represent scalar extremes for the concept '{column_name}'.
        Each pair should use descriptive phrases (e.g., 'completely drained', 'fully charged') rather than single words, to clearly represent the low and high ends of a scale.
        Format output as a Python list of tuples like this:
        [("completely drained", "fully charged"), ("unreliable performance", "dependable performance"), ...]
        Do not wrap the output in markdown or print statements.
        """
        content = self._call_openrouter(prompt)

        try:
            # Remove any markdown/code block wrapping
            content = re.sub(r"```(?:python)?\s*([\s\S]*?)```", r"\1", content, flags=re.IGNORECASE).strip()
            # Try to extract the list using eval
            pairs = eval(content)
            return [tuple(pair) for pair in pairs if isinstance(pair, (list, tuple)) and len(pair) == 2]
        except Exception as e:
            print("Failed to parse antonym pairs:", e)
            return []

    def rank_scales_with_llm(self, column_name: str, axis_to_emojis: dict) -> List[str]:
        options = "\n".join(
            f"Option {chr(65+i)} (axis: {a} → {b}): {' '.join(emojis)}"
            for i, ((a, b), emojis) in enumerate(axis_to_emojis.items())
        )
        print(options)
        prompt = f"""
        You are evaluating emoji scales for the concept '{column_name}'.
        Below are multiple emoji scales, each created from a semantic antonym axis:

        {options}

        Which one most accurately reflects a gradual transition from low to high for '{column_name}'?
        Reply with the best option letter only (e.g., A, B, C).
        """
        content = self._call_openrouter(prompt).upper()
        best = content.strip()
        idx = ord(best) - 65
        return list(axis_to_emojis.values())[idx] if 0 <= idx < len(axis_to_emojis) else []

    def generate_best_llm_scale(self, column_name: str, top_k=20, scale_size=3) -> List[str]:
        axes = self.get_llm_antonym_pairs(column_name)
        axis_to_emojis = {}
        for axis in axes:
            try:
                emoji_scale = self.generate_scale_with_axis(column_name, axis, top_k=top_k, scale_size=scale_size)
                axis_to_emojis[axis] = emoji_scale
            except Exception as e:
                print(f"Failed on axis {axis}: {e}")
        if not axis_to_emojis:
            return []
        return self.rank_scales_with_llm(column_name, axis_to_emojis)


# Example usage:
scaler = EmojiScaler(
    emoji_csv_path="../data/emoji_data_with_descriptions.csv",
    embedding_path="../data/emoji_description_embeddings.npy"
)

print("Best scale for 'battery':", scaler.generate_best_llm_scale("battery"))
print("Best scale for 'popularity':", scaler.generate_best_llm_scale("popularity"))

