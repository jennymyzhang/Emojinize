import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import requests
import re
from typing import List, Tuple
import ast

OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openai/gpt-4o-mini"

class EmojiScaler:
    def __init__(self, emoji_csv_path, embedding_path):
        self.emoji_df = pd.read_csv(emoji_csv_path)
        self.model = SentenceTransformer("paraphrase-mpnet-base-v2")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        description_embeddings = np.load(embedding_path)
        self.emoji_df["embedding"] = list(description_embeddings)

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().float().to(self.device)
        return torch.from_numpy(x).float().to(self.device)

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

    def get_llm_antonym_pairs_multi(self, column_name: str, num_queries=3) -> List[Tuple[str, str]]:
        all_pairs = []
        for _ in range(1):
            all_pairs.extend(self._get_llm_antonym_pairs_once(column_name))
        seen = set()
        deduped = []
        for a, b in all_pairs:
            key = (a.lower(), b.lower())
            if key not in seen:
                deduped.append((a, b))
                seen.add(key)
        return deduped

    def _get_llm_antonym_pairs_once(self, column_name: str) -> List[Tuple[str, str]]:
        prompt = f"""
        You are creating a semantic scale for the concept '{column_name}', using emojis to represent low and high levels of the concept.

        Your goal is to generate 10 pairs of phrases. Each pair must:
        - Represent **low vs. high** states of the concept.
        - Be **extremely concrete, literal, and visual**.
        - Use **direct, emotionally or physically expressive language** (e.g., "crying face", "heartbroken", "thumbs up").
        - Be **easily representable with standard emojis**, such that even a simple emoji-matching algorithm could assign appropriate emojis to each phrase.

        **Absolutely avoid**:
        - Metaphors, symbolic phrases, cultural references, or social situations.
        - Words like "forgotten", "quiet room", "city festival", "fading star", "hidden talent", etc.

        **Use phrases that are already similar to common emoji names or emoji alt-text**, such as:
        - "crying face"
        - "broken heart"
        - "red heart"
        - "thumbs down"
        - "smiling face"
        - "fire"
        - "ice cube"
        - "angry face"
        - "party popper"
        - "clapping hands"

        **Examples**:
        - For 'love': ("broken heart", "red heart")
        - For 'emotion': ("sad face", "smiling face")
        - For 'approval': ("thumbs down", "thumbs up")
        - For 'heat': ("ice cube", "fire")

        Now generate 10 such **low-high** phrase pairs for '{column_name}' using this approach.

        Return only a valid Python list of 2-tuples:
        [("low phrase", "high phrase"), ("...", "...")]
        """

        content = self._call_openrouter(prompt)
        print(content)
        try:
            if content.startswith("```"):
                content = re.sub(r"^```(?:python)?\n", "", content)
                content = re.sub(r"\n```$", "", content)
            parsed = ast.literal_eval(content)
            return [tuple(pair) for pair in parsed if isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(x, str) for x in pair)]
        except Exception as e:
            print("Failed to parse antonym pairs:", e)
            return []

    def filter_axes_by_similarity(self, axes: List[Tuple[str, str]], query: str, threshold=0.25):
        query_vec = self.model.encode(query, convert_to_tensor=True).to(self.device)
        good_axes = []
        for a, b in axes:
            a_vec = self.model.encode(a, convert_to_tensor=True).to(self.device)
            b_vec = self.model.encode(b, convert_to_tensor=True).to(self.device)
            sim_a = util.pytorch_cos_sim(query_vec, a_vec).item()
            sim_b = util.pytorch_cos_sim(query_vec, b_vec).item()
            if (sim_a + sim_b) / 2 > threshold:
                good_axes.append((a, b))
        return good_axes

    

    def generate_best_llm_scale(self, column_name: str, top_k=20, scale_size=3) -> List[str]:
        axes = self.get_llm_antonym_pairs_multi(column_name, num_queries=3)
        #axes = self.filter_axes_by_similarity(axes, column_name)
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

    def _closest_emoji(self, phrase: str, exclude: List[str] = []) -> str:
        phrase_vec = self.model.encode(phrase, convert_to_tensor=True).to(self.device)
        all_embeddings = torch.stack([
            torch.nn.functional.normalize(self._to_tensor(x), dim=0)
            for x in self.emoji_df["embedding"]
        ])
        similarities = util.pytorch_cos_sim(phrase_vec, all_embeddings)[0]

        # Mask out excluded emojis
        for i, emoji in enumerate(self.emoji_df["emoji"]):
            if emoji in exclude:
                similarities[i] = -float("inf")

        idx = similarities.argmax().item()
        return self.emoji_df.iloc[idx]["emoji"]


    def generate_scale_with_axis(self, column_name: str, axis_pair: Tuple[str, str], top_k=100, scale_size=5) -> List[str]:
        query_vec = self.model.encode(column_name, convert_to_tensor=True).to(self.device)
        a_vec = self.model.encode(axis_pair[0], convert_to_tensor=True).to(self.device)
        b_vec = self.model.encode(axis_pair[1], convert_to_tensor=True).to(self.device)

        a_vec = a_vec / (a_vec.norm() + 1e-6)
        b_vec = b_vec / (b_vec.norm() + 1e-6)
        axis_unit = (b_vec - a_vec) / ((b_vec - a_vec).norm() + 1e-6)
        midpoint = (a_vec + b_vec) / 2.0

        all_embeddings = torch.stack([
            torch.nn.functional.normalize(self._to_tensor(x), dim=0)
            for x in self.emoji_df["embedding"]
        ])
        similarities = util.pytorch_cos_sim(query_vec, all_embeddings)[0]
        top_indices = similarities.topk(top_k).indices.tolist()
        selected = self.emoji_df.iloc[top_indices].copy()

        selected_embeddings = torch.stack([
            torch.nn.functional.normalize(self._to_tensor(x), dim=0)
            for x in selected["embedding"]
        ])

        centered = selected_embeddings - midpoint
        projection = torch.matmul(centered, axis_unit)
        relevance = util.pytorch_cos_sim(query_vec, selected_embeddings)[0]
        semantic_boost = torch.cosine_similarity(selected_embeddings, b_vec.unsqueeze(0), dim=1)

        norm_proj = (projection - projection.mean()) / (projection.std() + 1e-6)
        norm_rel = (relevance - relevance.mean()) / (relevance.std() + 1e-6)
        norm_semantic_boost = (semantic_boost - semantic_boost.mean()) / (semantic_boost.std() + 1e-6)

        final_score = (0.5 * norm_proj + 0.3 * norm_rel + 0.2 * norm_semantic_boost)
        final_score_np = final_score.detach().cpu().numpy()
        selected["emoji_text"] = selected["emoji"] + " (" + np.round(final_score_np, 2).astype(str) + ")"
        selected["score"] = final_score_np.tolist()

        sorted_df = selected.sort_values(by="score", ascending=True).reset_index(drop=True)

        # ---- FORCED ENDPOINTS ----
        if len(sorted_df) < scale_size:
            return sorted_df["emoji_text"].tolist()

        low_emoji = self._closest_emoji(axis_pair[0])
        high_emoji = self._closest_emoji(axis_pair[1], exclude=[low_emoji])
        num_middle = scale_size - 2
        if num_middle > 0:
            indices = np.linspace(1, len(sorted_df) - 2, num=num_middle, dtype=int)
            middle = sorted_df.iloc[indices]["emoji"].tolist()
        else:
            middle = []

        return [low_emoji] + middle + [high_emoji]


    def rank_scales_with_llm(self, column_name: str, axis_to_emojis: dict) -> List[str]:
        options = "\n".join(
            f"Option {chr(65+i)} {' '.join(emojis)}"
            for i, (_, emojis) in enumerate(axis_to_emojis.items())
        )
        print(options)
        prompt = f"""
        Imagine you're designing a visual indicator using emojis for the concept '{column_name}'.
        Which of these best conveys a meaningful, expressive, and transition from low to high? Make sure the scale is understood by a wide audience.
        Dont include any special character emojis in the scale. And make sure people can associate the emojis with the concept '{column_name}'.

        {options}

        Reply with the best option letter only (e.g., A, B, C).
        """
        content = self._call_openrouter(prompt).upper().strip()
        idx = ord(content[0]) - 65 if content and content[0].isalpha() else -1
        return list(axis_to_emojis.values())[idx] if 0 <= idx < len(axis_to_emojis) else []


# Example usage:
scaler = EmojiScaler(
    emoji_csv_path="../data/emoji_data_with_descriptions.csv",
    embedding_path="../data/emoji_description_embeddings.npy"
)

print("Best scale for 'battery':", scaler.generate_best_llm_scale("battery"))
print("Best scale for 'popularity':", scaler.generate_best_llm_scale("popularity"))