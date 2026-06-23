"""Solver classico [A]+[B]: generazione candidati + scoring pesato (stile Sangati).

Per ogni candidato c e ciascun indizio h_i si misura una forza di associazione:
  - MWE:        numero di espressioni (polirematiche/proverbi) in cui c e h_i co-occorrono
  - embedding:  similarità coseno cos(c, h_i)  (se disponibili gli embeddings)

Un indizio è "coperto" da c se c'è associazione MWE > 0 oppure similarità embedding
sopra una soglia. Lo score premia prima di tutto la COPERTURA (quanti dei 5 indizi
sono connessi a c) e poi la forza complessiva:

  score(c) = COVERAGE_WEIGHT * copertura
           + str_mwe (numero totale di co-occorrenze)
           + beta * somma delle similarità embedding

Così un candidato legato a tutti e 5 gli indizi batte uno legato solo a 2, come nel
sistema UNIOR4NLP/"Il Mago della Ghigliottina" (Sangati et al.).
"""
from __future__ import annotations

from dataclasses import dataclass

from resources import LexicalResources
from embeddings import EmbeddingModel

COVERAGE_WEIGHT = 1000.0  # la copertura domina lo score (ordinamento lessicografico)
EMB_SIM_THRESHOLD = 0.35  # soglia oltre la quale un embedding "copre" un indizio


@dataclass
class Prediction:
    solution: str
    score: float
    ranked: list[tuple[str, float]]  # top candidati con punteggio


class ClassicSolver:
    def __init__(
        self,
        resources: LexicalResources,
        embeddings: EmbeddingModel | None = None,
        alpha: float = 1.0,          # mantenuto per compatibilità (non più usato)
        beta: float = 2.0,           # peso della componente embedding
        emb_neighbors: int = 30,
        emb_threshold: float = EMB_SIM_THRESHOLD,
        scoring: str = "coverage",   # "coverage" (rigido) oppure "assoc" (PMI, soft)
        corpus_pmi_threshold: float = 2.0,  # PMI minimo perché il corpus "copra" un indizio
        gamma: float = 0.3,                 # peso del contributo corpus nello score
    ):
        self.res = resources
        self.emb = embeddings
        self.alpha = alpha
        self.beta = beta
        self.emb_neighbors = emb_neighbors
        self.emb_threshold = emb_threshold
        self.scoring = scoring
        self.corpus_pmi_threshold = corpus_pmi_threshold
        self.gamma = gamma

    def solve(self, hints: list[str], top_k: int = 50) -> Prediction:
        hints_l = [h.lower() for h in hints]
        hint_set = set(hints_l)
        # forza MWE per indizio: cooc[h] è Counter(parola -> n. espressioni condivise)
        clue_coocs = [self.res.cooc.get(h, {}) for h in hints_l]
        # associazioni da corpus per indizio: {neighbor: pmi}
        clue_corpus = [self.res.corpus_assoc.get(h, {}) for h in hints_l]

        # [A] pool candidati: parole che co-occorrono con almeno un indizio...
        pool: set[str] = set()
        for cc in clue_coocs:
            pool.update(cc.keys())
        for ca in clue_corpus:           # ...e i collocati da corpus
            pool.update(ca.keys())
        # ...espanso con i vicini semantici degli indizi (alza il recall)
        if self.emb is not None:
            for h in hints_l:
                for w, _sim in self.emb.neighbors(h, topn=self.emb_neighbors):
                    if len(w) > 2 and "_" not in w:
                        pool.add(w)
        pool -= hint_set  # un candidato non può essere un indizio
        if not pool:
            return Prediction(solution="", score=0.0, ranked=[])

        # [B] scoring pesato con copertura dei 5 indizi.
        # La forza MWE è normalizzata per la rarità del candidato (stile PMI/IDF):
        # una parola specifica vale più di una generica che co-occorre ovunque.
        import math

        # frequenza (n. espressioni) di ciascun indizio, per la pesatura PMI
        clue_freq = [max(1, len(self.res.word_to_exprs.get(h, ()))) for h in hints_l]

        scored: list[tuple[str, float]] = []
        for c in pool:
            mwe_counts = [cc.get(c, 0) for cc in clue_coocs]
            corpus_pmi = [ca.get(c, 0.0) for ca in clue_corpus]
            freq_c = max(1, len(self.res.word_to_exprs.get(c, ())))
            sims = (
                [max(0.0, self.emb.similarity(c, h)) for h in hints_l]
                if self.emb is not None else [0.0] * len(hints_l)
            )
            if self.scoring == "assoc":
                # PMI-like: ogni indizio contribuisce con la forza SPECIFICA della
                # connessione, normalizzata per la rarità di candidato e indizio.
                # Niente copertura rigida: connessioni forti/specifiche emergono
                # anche se il candidato copre meno indizi-spazzatura.
                score = 0.0
                for i in range(len(hints_l)):
                    if mwe_counts[i] > 0:
                        score += mwe_counts[i] / math.sqrt(freq_c * clue_freq[i])
                    score += self.beta * sims[i]
            else:
                # copertura rigida: un indizio è coperto se c'è una MWE, una
                # collocazione di corpus con PMI sopra soglia, o un embedding simile
                coverage = sum(
                    1 for i in range(len(hints_l))
                    if mwe_counts[i] > 0
                    or corpus_pmi[i] >= self.corpus_pmi_threshold
                    or sims[i] >= self.emb_threshold
                )
                str_mwe = sum(mwe_counts) / math.sqrt(freq_c)
                score = (COVERAGE_WEIGHT * coverage + str_mwe
                         + self.beta * sum(sims)
                         + self.gamma * sum(corpus_pmi))
            scored.append((c, score))

        # ordinamento deterministico: punteggio decrescente, poi parola alfabetica
        scored.sort(key=lambda x: (-x[1], x[0]))
        best, best_score = scored[0]
        return Prediction(solution=best, score=best_score, ranked=scored[:top_k])

    def candidate_pool(self, hints: list[str], top_k: int) -> list[str]:
        """Lista dei top-k candidati da passare all'LLM nella fase [C]."""
        return [w for w, _ in self.solve(hints, top_k=top_k).ranked]
