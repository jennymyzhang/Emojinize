import pandas as pd
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Initialize sentiment analysis pipeline (HuggingFace)
sentiment_pipeline = pipeline("sentiment-analysis")

# Load data and embeddings
df = pd.read_csv('emoji_data_with_keywords.csv')
description_embeddings = np.load('emoji_description_embeddings.npy')
keyword_embeddings = np.load('emoji_keyword_embeddings.npy')



def compute_and_save_emoji_sentiments(df, sentiment_pipeline, output_file='emoji_data_with_sentiment.csv'):
    sentiments = []
    
    for desc in df['description']:
        sentiment_result = sentiment_pipeline(desc)[0]
        sentiments.append({
            'label': sentiment_result['label'],  # POSITIVE / NEGATIVE / NEUTRAL
            'score': sentiment_result['score']   # Confidence score
        })
    
    df['sentiment_label'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]
    
    df.to_csv(output_file, index=False)
    print(f"Sentiment labels added and saved to {output_file}")
    
    return df

compute_and_save_emoji_sentiments(df, sentiment_pipeline, 'emoji_data_with_sentiment.csv')