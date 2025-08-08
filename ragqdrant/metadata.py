from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from langdetect import detect as detect_lang
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None  # type: ignore


def detect_language(text: str) -> Optional[str]:
    try:
        return detect_lang(text)
    except Exception:
        return None


def simple_keywords(texts: List[str], top_k: int = 10) -> List[List[str]]:
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    results: List[List[str]] = []
    for row in X:
        row = row.tocoo()
        scores = list(zip(row.col, row.data))
        scores.sort(key=lambda x: x[1], reverse=True)
        terms = [feature_names[i] for i, _ in scores[:top_k]]
        results.append(terms)
    return results


def optional_spacy_ner(texts: List[str], model_name: str) -> List[Dict[str, List[str]]]:
    if spacy is None:
        return [defaultdict(list) for _ in texts]
    try:
        nlp = spacy.load(model_name)  # type: ignore
    except Exception:
        return [defaultdict(list) for _ in texts]
    entities_list: List[Dict[str, List[str]]] = []
    for t in texts:
        doc = nlp(t)
        ent_map: Dict[str, List[str]] = defaultdict(list)
        for ent in doc.ents:
            ent_map[ent.label_].append(ent.text)
        entities_list.append(ent_map)
    return entities_list


def readability_score(text: str) -> float:
    # Simple heuristic: inverse of average sentence length
    sentences = [s for s in text.split(".") if s.strip()]
    if not sentences:
        return 0.0
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    score = 100.0 / (1.0 + avg_len)
    return round(score, 3)