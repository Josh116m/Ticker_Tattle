import os
from typing import Optional

import pandas as pd

ART_DIR = os.path.join('artifacts','phase2')


def run_finbert(df_clean: pd.DataFrame, text_col: str = 'title', use_gpu: bool = True) -> pd.DataFrame:
    # Uses HuggingFace transformers pipeline; fallback to CPU if CUDA not available
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    model_name = 'ProsusAI/finbert'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe = pipeline('text-classification', model=model, tokenizer=tokenizer, device=0 if use_gpu else -1, truncation=True)

    texts = (df_clean[text_col].fillna('') + ' ' + df_clean['description'].fillna('')).str.strip().tolist()
    outputs = pipe(texts, return_all_scores=True, truncation=True)
    # Compute polarity = p_pos - p_neg; confidence = max prob
    polarities = []
    confidences = []
    for scores in outputs:
        # FinBERT labels: Positive, Negative, Neutral (case-insensitive)
        d = {s['label'].lower(): float(s['score']) for s in scores}
        p_pos = d.get('positive', 0.0)
        p_neg = d.get('negative', 0.0)
        conf = max(d.values()) if d else 0.0
        polarities.append(p_pos - p_neg)
        confidences.append(conf)
    out = pd.DataFrame({
        'article_id': df_clean['article_id'].values,
        'polarity': polarities,
        'confidence': confidences,
    })
    return out


def write_article_sentiment(df_sent: pd.DataFrame) -> str:
    os.makedirs(ART_DIR, exist_ok=True)
    path = os.path.join(ART_DIR, 'article_sentiment.parquet')
    df_sent.to_parquet(path, index=False)
    return path

