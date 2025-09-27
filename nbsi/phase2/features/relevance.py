import os
from typing import Dict, List

import numpy as np
import pandas as pd

ART_DIR = os.path.join('artifacts','phase2')

SECTOR_PROMPTS: Dict[str, str] = {
    'XLK': 'technology, software, semiconductors, cloud, AI, hardware, IT services',
    'XLF': 'banks, insurance, capital markets, financial services, brokerage, fintech',
    'XLY': 'consumer discretionary, retail, autos, travel, entertainment, apparel',
    'XLV': 'healthcare, pharmaceuticals, biotechnology, medical devices, services',
    'XLE': 'energy, oil, gas, exploration, production, refining, midstream, services',
    'XLB': 'materials, chemicals, mining, metals, packaging, construction materials',
    'XLI': 'industrials, aerospace, defense, machinery, transportation, logistics',
    'XLU': 'utilities, electricity, gas, water, power generation, transmission',
    'XLRE': 'real estate, REITs, property management, commercial, residential',
    'XLC': 'communication services, media, telecom, internet content, advertising',
    'XLP': 'consumer staples, food, beverage, household products, personal care',
}


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / n


def compute_relevance(df_clean: pd.DataFrame, use_gpu: bool = True) -> pd.DataFrame:
    # Sentence-Transformers model; compatible tokenizer/model hub
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda' if use_gpu else 'cpu')
    texts = (df_clean['title'].fillna('') + ' ' + df_clean['description'].fillna('')).str.strip().tolist()
    art_emb = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    prompts = list(SECTOR_PROMPTS.values())
    prompt_keys = list(SECTOR_PROMPTS.keys())
    prm_emb = model.encode(prompts, normalize_embeddings=True, batch_size=len(prompts), show_progress_bar=False)

    # Cosine similarities in [-1,1] already due to normalization
    sims = np.matmul(art_emb, prm_emb.T)
    # Scale to [0,1]
    sims01 = (sims + 1.0) * 0.5

    out = pd.DataFrame({'article_id': df_clean['article_id'].values})
    for i, key in enumerate(prompt_keys):
        out[f'relevance_{key}'] = sims01[:, i]
    return out


def write_article_relevance(df_rel: pd.DataFrame) -> str:
    os.makedirs(ART_DIR, exist_ok=True)
    path = os.path.join(ART_DIR, 'article_relevance.parquet')
    df_rel.to_parquet(path, index=False)
    return path

