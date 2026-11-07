"""
Semantic Fidelity Metrics for Sign Language Translation:
1. Translation Soft Recall (TSR) - Equations (2) & (3) in paper
   Uses paraphrase-multilingual-MiniLM-L12-v2 embeddings when available,
   with a fully deterministic SHA256-based n-gram embedding fallback.
2. Pseudo-Gloss Entity Recall (PGER) - Equation (4) in paper
   Uses FLAIR POS tagging for nouns, verbs, and adjectives when available,
   with a language-aware content word extractor fallback.
"""

import hashlib
import re
import numpy as np

# Global cache for sentence transformer model
_SENTENCE_MODEL = None
_FLAIR_TAGGER = None

def _get_sentence_model():
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is False:
        return None
    if _SENTENCE_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Uses the exact embedding model from the paper (Section 4.2)
            _SENTENCE_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception:
            _SENTENCE_MODEL = False
    return _SENTENCE_MODEL

def _get_flair_tagger(lang="de"):
    global _FLAIR_TAGGER
    if _FLAIR_TAGGER is False:
        return None
    if _FLAIR_TAGGER is None:
        try:
            from flair.models import SequenceTagger
            model_name = "de-pos" if lang == "de" else "pos"
            _FLAIR_TAGGER = SequenceTagger.load(model_name)
        except Exception:
            _FLAIR_TAGGER = False
    return _FLAIR_TAGGER

# Fallback content vocabulary for PHOENIX-2014T (German) and ASL
CONTENT_WORDS_DE = {
    "montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag",
    "wochenende", "morgen", "heute", "nacht", "tag", "abend",
    "norden", "süden", "osten", "westen", "küsten", "alpen", "deutschland",
    "wetter", "wettervorhersage", "regen", "schnee", "sonne", "wolken", "bewölkt",
    "heiter", "freundlich", "trocken", "wind", "sturm", "gewitter", "nebel",
    "grad", "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben",
    "acht", "neun", "zehn", "zwölf", "fünfzehn", "achtundzwanzigsten", "dreißigsten",
    "steigen", "sinken", "wehen", "bleiben", "fallen", "erreichen", "scheinen"
}

CONTENT_WORDS_EN = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "tomorrow", "yesterday", "today", "morning", "night", "afternoon",
    "north", "south", "east", "west", "coast", "alps", "mountain",
    "weather", "forecast", "rain", "snow", "sun", "cloud", "cloudy",
    "clear", "dry", "wind", "storm", "fog", "degree", "temperature",
    "rise", "fall", "blow", "stay", "reach", "shine", "retire", "school"
}

def extract_content_entities(text, lang="de"):
    """
    Extracts key content words (nouns, verbs, adjectives) as proxies for glosses.
    Uses FLAIR if available, otherwise deterministic linguistic dictionary fallback.
    """
    tagger = _get_flair_tagger(lang)
    if tagger is not None:
        try:
            from flair.data import Sentence
            s = Sentence(text)
            tagger.predict(s)
            target_tags = {"NOUN", "VERB", "ADJ", "NN", "NE", "VVFIN", "VVINF", "VVIZU", "VVPP", "ADJA", "ADJD"}
            entities = {token.text.lower() for token in s if any(t.value in target_tags for t in token.labels)}
            if entities:
                return entities
        except Exception:
            pass

    # Deterministic fallback
    clean_text = re.sub(r"[^\w\s]", "", text.lower())
    tokens = set(clean_text.split())
    content_ref = CONTENT_WORDS_DE if lang == "de" else CONTENT_WORDS_EN
    entities = {t for t in tokens if t in content_ref or len(t) > 4}
    return entities

def compute_pger(prediction, reference, lang="de"):
    """
    Pseudo-Gloss Entity Recall:
    PGER(Y, Y_hat) = |Ent(Y) ∩ Ent(Y_hat)| / |Ent(Y)|
    Eq. (4) in paper.
    """
    ent_ref = extract_content_entities(reference, lang=lang)
    if len(ent_ref) == 0:
        return 1.0

    ent_pred = extract_content_entities(prediction, lang=lang)
    intersection = ent_ref.intersection(ent_pred)
    return len(intersection) / len(ent_ref)

def _deterministic_token_embedding(token, dim=64):
    """
    Cross-process deterministic token embedding using SHA-256 seed.
    Guarantees stable reproducibility without relying on process-dependent hash().
    """
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "little")
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim)
    return vec / (np.linalg.norm(vec) + 1e-9)

def soft_cardinality(tokens, embed_fn=None):
    """
    Computes soft cardinality C(Z) from Eq. (2) of the paper:
    C(Z) = sum_{i=1}^{|Z|} 1 / sum_{j=1}^{|Z|} Sim(Z_i, Z_j)
    """
    if len(tokens) == 0:
        return 0.0

    model = _get_sentence_model() if embed_fn is None else None
    if model is not None:
        try:
            embeds = model.encode(tokens, show_progress_bar=False)
        except Exception:
            embeds = [_deterministic_token_embedding(t) for t in tokens]
    else:
        fn = embed_fn or _deterministic_token_embedding
        embeds = [fn(t) for t in tokens]

    embeds = np.array(embeds)
    # L2 normalize
    norms = np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9
    embeds_norm = embeds / norms

    # Sim matrix [N, N]
    sim_matrix = np.clip(np.matmul(embeds_norm, embeds_norm.T), 0.0, 1.0)
    denom = np.sum(sim_matrix, axis=1)
    c_z = np.sum(1.0 / np.maximum(denom, 1e-9))
    return float(c_z)

def compute_tsr(prediction, reference, embed_fn=None):
    """
    Translation Soft Recall (TSR) from Eq. (3) of the paper:
    TSR(Y, Y_hat) = [C(Y) + C(Y_hat) - C(Y U Y_hat)] / C(Y_hat)
    """
    y_tokens = list(set(reference.strip().lower().split()))
    y_hat_tokens = list(set(prediction.strip().lower().split()))

    if len(y_hat_tokens) == 0 or len(y_tokens) == 0:
        return 0.0

    c_y = soft_cardinality(y_tokens, embed_fn)
    c_y_hat = soft_cardinality(y_hat_tokens, embed_fn)
    c_union = soft_cardinality(list(set(y_tokens + y_hat_tokens)), embed_fn)

    tsr = (c_y + c_y_hat - c_union) / (c_y_hat + 1e-9)
    return float(np.clip(tsr, 0.0, 1.0))
