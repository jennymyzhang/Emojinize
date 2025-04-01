import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Step 1: Load the emoji CSV
file_path = '../data/emojis.csv'
df = pd.read_csv(file_path)

# Confirm column names
print(df.head())  # You can remove this after confirmation

# Step 2: Extract descriptions (edit this if your column is different)
descriptions = df['description'].astype(str).tolist()

# Step 3: Initialize sentence transformer model
model = SentenceTransformer('paraphrase-mpnet-base-v2')

# Step 4: Generate embeddings for descriptions
description_embeddings = model.encode(descriptions, convert_to_numpy=True)

# Step 5: Save embeddings and DataFrame to files
# Save embeddings to a .npy file
np.save('emoji_description_embeddings.npy', description_embeddings)

# Save the dataframe (optional, if you modify it)
df.to_csv('emoji_data_with_descriptions.csv', index=False)

print("Embeddings and data saved successfully!")
