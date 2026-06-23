"""Solver a grafo con spreading activation (Personalized PageRank).

Idea (stile OTTHO / UNIOR4NLP, e dei "grafi pesati" usati dai migliori sistemi):
costruiamo, per ogni partita, un grafo locale di associazioni tra parole, con archi
pesati da due fonti:
  - co-occorrenza nelle polirematiche/proverbi (forza MWE, normalizzata per rarità);
  - similarità coseno tra embedding (associazione distribuzionale).
Poi propaghiamo attivazione partendo da CIASCUN indizio (Personalized PageRank) e
combiniamo i cinque profili di attivazione in modo da premiare le parole raggiunte
da TUTTI e cinque gli indizi (somma dei logaritmi). Questo sfrutta cammini multi-hop
(indizio → parola intermedia → soluzione) che lo scoring diretto non vede.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from resources import LexicalResources
from embeddings import EmbeddingModel


@dataclass
class Prediction:
    solution: str
    score: float
    ranked: list[tuple[str, float]]


class GraphSolver:
    def __init__(
        self,
        resources: LexicalResources,
        embeddings: EmbeddingModel,
        emb_neighbors: int = 60,
        tau: float = 0.30,        # soglia di similarità per creare un arco embedding
        mwe_weight: float = 1.0,  # peso degli archi MWE
        emb_weight: float = 1.0,  # peso degli archi embedding
        restart: float = 0.40,    # probabilità di restart del random walk
        iters: int = 40,
    ):
        self.res = resources
        self.emb = embeddings
        self.emb_neighbors = emb_neighbors
        self.tau = tau
        self.mwe_weight = mwe_weight
        self.emb_weight = emb_weight
        self.restart = restart
        self.iters = iters

    def _build_vocab(self, clues: list[str]) -> list[str]:
        V: set[str] = set(clues)
        for h in clues:
            # vicini MWE (co-occorrenza diretta)
            V.update(self.res.cooc.get(h, {}).keys())
            # vicini distribuzionali
            for w, _s in self.emb.neighbors(h, topn=self.emb_neighbors):
                if len(w) > 2 and "_" not in w:
                    V.add(w)
        return list(V)

    def _adjacency(self, V: list[str], clues: set[str]) -> np.ndarray:
        n = len(V)
        idx = {w: i for i, w in enumerate(V)}

        # --- archi embedding: matrice coseno soglia ---
        dim = self.emb.kv.vector_size
        X = np.zeros((n, dim), dtype=np.float32)
        for i, w in enumerate(V):
            v = self.emb.vec(w)
            if v is not None:
                X[i] = v
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        cos = Xn @ Xn.T
        W = self.emb_weight * np.maximum(0.0, cos - self.tau)

        # --- archi MWE (sparsi) ---
        for w in V:
            i = idx[w]
            freq_w = max(1, len(self.res.word_to_exprs.get(w, ())))
            for v, cnt in self.res.cooc.get(w, {}).items():
                j = idx.get(v)
                if j is not None and j != i:
                    wgt = self.mwe_weight * cnt / math.sqrt(freq_w)
                    W[i, j] += wgt
                    W[j, i] += wgt

        np.fill_diagonal(W, 0.0)
        return W

    def solve(self, hints: list[str], top_k: int = 50) -> Prediction:
        clues = [h.lower() for h in hints]
        clue_set = set(clues)
        V = self._build_vocab(clues)
        if not V:
            return Prediction("", 0.0, [])
        idx = {w: i for i, w in enumerate(V)}
        n = len(V)
        W = self._adjacency(V, clue_set)

        # matrice di transizione (normalizzazione per colonne)
        col = W.sum(axis=0)
        col[col == 0] = 1.0
        T = W / col

        # Personalized PageRank da ciascun indizio presente nel grafo
        seeds = [idx[c] for c in clues if c in idx]
        if not seeds:
            return Prediction("", 0.0, [])
        log_acc = np.zeros(n, dtype=np.float64)
        eps = 1e-9
        for s in seeds:
            e = np.zeros(n, dtype=np.float64)
            e[s] = 1.0
            p = e.copy()
            for _ in range(self.iters):
                p = (1 - self.restart) * (T @ p) + self.restart * e
            log_acc += np.log(p + eps)  # somma dei log => premia attivazione da TUTTI

        order = np.argsort(-log_acc)
        ranked = [(V[i], float(log_acc[i])) for i in order if V[i] not in clue_set]
        if not ranked:
            return Prediction("", 0.0, [])
        return Prediction(ranked[0][0], ranked[0][1], ranked[:top_k])

    def candidate_pool(self, hints: list[str], top_k: int) -> list[str]:
        return [w for w, _ in self.solve(hints, top_k=top_k).ranked]
