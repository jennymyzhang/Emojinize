import sys
import torch
import spacy
import logging
from typing import Dict, Any, List, Tuple
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util


class AspectExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_aspects(self, text: str) -> List[str]:
        doc = self.nlp(text)
        aspects = set()

        for chunk in doc.noun_chunks:
            cleaned = chunk.text.strip().lower()
            if len(cleaned) > 2:
                aspects.add(cleaned)

        for token in doc:
            if token.dep_ == "compound" and token.head:
                compound_phrase = f"{token.text} {token.head.text}".lower()
                aspects.add(compound_phrase)

        return list(aspects)


class AspectSentimentAnalyzer:
    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        self.pipeline = pipeline("sentiment-analysis", model=model_name)

    def analyze_aspect(self, aspect: str) -> Tuple[str, float]:
        try:
            results = self.pipeline(aspect)
            if results:
                return results[0]['label'], float(results[0]['score'])
        except Exception:
            return "neutral", 0.0
        return "neutral", 0.0

    def analyze_aspects(self, aspects: List[str]) -> Dict[str, Tuple[str, float]]:
        return {aspect: self.analyze_aspect(aspect) for aspect in aspects}


class CoreSentimentAnalyzer:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.pipeline = pipeline("sentiment-analysis", model=model_name)

    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        try:
            results = self.pipeline(text)
            if results:
                return results[0]['label'], float(results[0]['score'])
        except Exception:
            return "neutral", 0.0
        return "neutral", 0.0


class TrendDetector:
    def __init__(self):
        self.embedding_model = SentenceTransformer('paraphrase-mpnet-base-v2')

        self.increase_refs = [
            "growth", "rising", "surging", "expansion", "bullish", "increasing", "upward", "ascending",
            "accelerating", "improving", "soaring", "boosting", "uptrend"
        ]
        self.decrease_refs = [
            "decline", "falling", "shrinking", "reduction", "bearish", "decreasing", "downward", "descending", "dropping",
            "plummeting", "slumping", "worsening", "collapsing", "downtrend"
        ]

        self.increase_embeddings = self.embedding_model.encode(self.increase_refs, convert_to_tensor=True)
        self.decrease_embeddings = self.embedding_model.encode(self.decrease_refs, convert_to_tensor=True)
        self.threshold = 0.3

    def detect_trend(self, text: str) -> Tuple[str, str]:
        text_embedding = self.embedding_model.encode(text, convert_to_tensor=True)

        inc_similarities = util.pytorch_cos_sim(text_embedding, self.increase_embeddings).max().item()
        dec_similarities = util.pytorch_cos_sim(text_embedding, self.decrease_embeddings).max().item()

        if inc_similarities > self.threshold and inc_similarities > dec_similarities:
            return "Increase", "📈"
        elif dec_similarities > self.threshold and dec_similarities > inc_similarities:
            return "Decrease", "📉"
        else:
            return "Neutral", "↔"


if __name__ == "__main__":
    aspect_extractor = AspectExtractor()
    aspect_sentiment_analyzer = AspectSentimentAnalyzer()
    sentiment_analyzer = CoreSentimentAnalyzer()
    trend_detector = TrendDetector()

    sample_texts = [
        "The stock prices are surging, and profits are increasing rapidly!",
        "Our revenue has been in a sharp decline this quarter.",
        "New investments are boosting our market presence.",
    ]

    for text in sample_texts:
        print("=" * 80)
        print(f"Text: {text}")

        # Extract aspects
        aspects = aspect_extractor.extract_aspects(text)
        print(f"Extracted Aspects: {aspects}")

        # Perform aspect-based sentiment analysis
        aspect_sentiments = aspect_sentiment_analyzer.analyze_aspects(aspects)
        print(f"Aspect-Based Sentiment: {aspect_sentiments}")

        # Perform core sentiment analysis
        sentiment, confidence = sentiment_analyzer.analyze_sentiment(text)
        print(f"Overall Sentiment: {sentiment} (Confidence: {confidence:.4f})")

        # Detect trend
        trend, emoji = trend_detector.detect_trend(text)
        print(f"Trend Detected: {trend} {emoji}\n")
