import nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer("paraphrase-mpnet-base-v2")

def encode(text: str):
    return model.encode(text)

def cosine_sim(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def get_best_sense_by_embedding(context: str, target_word: str, threshold: float = 0.1):
    """
    Disambiguate a word using WordNet glosses and context.
    Handles:
    - No synsets
    - Multiple synsets but no context
    - Valid disambiguation with score threshold
    """
    synsets = wn.synsets(target_word.replace(" ", "_"))
    
    if not synsets:
        return {
            "word": target_word,
            "best_sense": None,
            "definition": "No synsets found.",
            "examples": [],
            "score": 0
        }

    # No or meaningless context
    if not context or len(context.split()) < 2:
        if len(synsets) > 1:
            return "Need more context to disambiguate."
        else:
            # Only one meaning, so return it
            syn = synsets[0]
            return {
                "word": target_word,
                "best_sense": syn.name(),
                "definition": syn.definition(),
                "examples": syn.examples(),
                "score": 1.0
            }

    context_embedding = encode(context)
    best_score = -1
    best_synset = None

    for syn in synsets:
        gloss = syn.definition() + " " + " ".join(syn.examples())
        gloss_embedding = encode(gloss)
        score = cosine_sim(context_embedding, gloss_embedding)

        if score > best_score:
            best_score = score
            best_synset = syn

    if best_score < threshold:
        return "Need more description please!"

    return {
        "word": target_word,
        "best_sense": best_synset.name(),
        "definition": best_synset.definition(),
        "examples": best_synset.examples(),
        "score": best_score
    }
    
