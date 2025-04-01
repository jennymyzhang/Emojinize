# Emoji Recommendation Tool
---
## Setup

### Install requirements

```bash
pip install -r requirements.txt
```
## Usage

```bash
cd src

python3 -m emoji_data.generate_emojis \
  --description "This table contains information about financial and environmental indicators." \
  --column_file ../columns.txt
