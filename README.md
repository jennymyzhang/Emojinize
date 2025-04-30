# 🌐 Emoji Annotation Toolkit

This project generates emoji-based annotations for tabular data, including:

- **Numerical columns**: Semantic emoji scales (e.g., 🔵 ➡️ 🔴)
- **Categorical values**: Emoji representations (e.g., “Airport” → ✈️)
- **Column headers**: Emoji descriptors (e.g., “Price” → 💰📈)

---
## Setup

### Install requirements

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

### 1. Annotate a CSV with emojis

```bash
python generate_emojis.py \
  --input_csv ../data/my_table.csv \
  --output_json ../results/annotated.json \
  --top_k 5

The input CSV should have:

- **Top-left cell**: table description  
- **2nd row**: column headers  
- **From 3rd row**: data rows  


## 🧠 Core Functionality Overview

### 1. `generate_emoji_annotation_json()`
**File:** `generate_emojis.py`  
**Purpose:** Full pipeline to annotate a CSV with:
- Emoji scales for numeric columns  
- Emoji matches for categorical values  
- Emoji representations for column names  

#### How to Run:
```bash
python generate_emojis.py \
  --input_csv ../data/my_table.csv \
  --output_json ../results/annotated.json \
  --top_k 5

### 2. generate_best_llm_scale(column_name)
**File: ** `generate_scale.py`
**Purpose: **
Uses OpenRouter to generate antonym phrase pairs → projects emojis along semantic axis → selects and ranks emoji scales.

#### How to Run (Python example):
```bash
from scale.generate_scale import EmojiScaler
scaler = EmojiScaler()
scales = scaler.generate_best_llm_scale("Temperature")
print(scales)

### 3. `find_emojis_for_each_keyword()`
**File:** `generate_emoji_via_embedding.py`  
**Purpose:**  
Matches a query string to relevant emojis using:
- OpenRouter to rewrite the query into an expressive, emoji-matchable sentence
- KeyBERT to extract keywords from that rewritten sentence
- Cosine similarity to rank emojis using description and keyword embeddings

**Used for:**
- Categorical values
- Column names
- Manual queries (via CLI or notebook)

---

#### 📥 Inputs
- `query_text`: A string formatted as  
  `"Amazon in the column of: Company and table description: E-Commerce"`
- `df`: DataFrame of emoji metadata
- `description_embeddings`: `np.ndarray` of precomputed sentence embeddings for emoji descriptions
- `keyword_embeddings`: `np.ndarray` of precomputed sentence embeddings for emoji keywords
- `embedding_model`: a `SentenceTransformer` model
- `keyword_model`: a `KeyBERT` model
- `desc_weight` (`float`): weight for description embedding match
- `keyword_weight` (`float`): weight for keyword embedding match
- `top_k` (`int`): number of top results to return

---

#### 📤 Output
A dictionary mapping the LLM-enhanced description to a list of emoji match dicts:

```python
{
  "Amazon is a company known for delivery boxes, warehouses, and online shopping": [
    {"emoji": "📦", "similarity": 0.83, "description": "...", "keywords": "..."},
    {"emoji": "🚚", "similarity": 0.77, ...},
    ...
  ]
}


#### Run From CLI

```bash
python generate_emoji_via_embedding.py \
  --query "Amazon in the column of: Company and table description: E-Commerce" \
  --top_k 5 \
  --desc_weight 0.2 \
  --keyword_weight 0.8

