"""Wrapper sugli embeddings italiani (word2vec / fastText) via gensim.

Gestisce con eleganza l'assenza del modello: se il file non è disponibile,
`EmbeddingModel.load()` restituisce None e il solver ricade sul solo segnale
delle polirematiche.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class EmbeddingModel:
    def __init__(self, kv):
        self.kv = kv  # gensim KeyedVectors

    @classmethod
    def load(cls, path: str | Path, limit: int | None = None) -> "EmbeddingModel | None":
        """Carica gli embeddings.

        - file .bin (fastText Facebook): qualità migliore, gestisce OOV via subword.
        - file .vec/.txt (formato word2vec testuale): per restare leggeri si possono
          caricare solo le prime `limit` parole (le più frequenti), risparmiando RAM
          e tempo. `limit` è preso da IT_EMBEDDINGS_LIMIT se non passato.
        """
        path = str(path or "")
        if not path or not Path(path).exists():
            return None
        from gensim.models import KeyedVectors

        if path.endswith(".bin") and not path.endswith(".vec.bin"):
            import gensim.models.fasttext as ft
            return cls(ft.load_facebook_vectors(path))

        if limit is None:
            import os
            lim = os.environ.get("IT_EMBEDDINGS_LIMIT", "")
            limit = int(lim) if lim.isdigit() else None
        kv = KeyedVectors.load_word2vec_format(path, binary=False, limit=limit)
        return cls(kv)

    def similarity(self, a: str, b: str) -> float:
        try:
            return float(self.kv.similarity(a.lower(), b.lower()))
        except KeyError:
            return 0.0

    def vec(self, w: str) -> np.ndarray | None:
        try:
            return self.kv.get_vector(w.lower())
        except KeyError:
            return None

    def neighbors(self, word: str, topn: int = 30) -> list[tuple[str, float]]:
        """Parole più simili (vicini semantici) usate per espandere i candidati."""
        try:
            return [
                (w.lower(), float(s))
                for w, s in self.kv.most_similar(word.lower(), topn=topn)
            ]
        except KeyError:
            return []

    def aggregate_similarity(self, candidate: str, hints: list[str]) -> float:
        """Similarità aggregata candidato↔indizi.

        Usa la media delle similarità coseno; gli indizi fuori vocabolario
        contribuiscono 0. La media (non la somma) evita di favorire candidati
        legati a un solo indizio.
        """
        sims = [self.similarity(candidate, h) for h in hints]
        return float(np.mean(sims)) if sims else 0.0
