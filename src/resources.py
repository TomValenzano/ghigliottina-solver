"""Risorse lessicali: polirematiche, proverbi e indice di co-occorrenza.

Fornisce:
  - caricamento e normalizzazione delle liste (polirematiche, demauro.poli, proverbi)
  - indice di co-occorrenza parola -> parole che compaiono nella stessa espressione
  - generazione candidati a partire dai 5 indizi (stile OTTHO)
  - bonus di associazione (quanti indizi formano un'espressione nota col candidato)
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

# Stopword italiane + articoli/preposizioni che non sono parole-contenuto utili
STOPWORDS = set(
    "a al alla allo ai agli alle del dello della dei degli delle di da in con su per "
    "tra fra e o ed od il lo la i gli le un uno una un d l c se non ne ci vi mi "
    "ti si che chi cui come piu più meno ma anche solo sul sui sulla sullo sue suo "
    "nel nei nella nello negli dal dai dalla dallo dagli col coi nello è ho ha hai "
    "questo questa quello quella loro suoi sua mio mia tuo tua suoi più meno ogni "
    "essere avere fare dei delle".split()
)

_TOKEN_RE = re.compile(r"[a-zàèéìòùA-ZÀÈÉÌÒÙ]+")
_HAS_LETTER = re.compile(r"[a-zàèéìòù]", re.IGNORECASE)


def _tokenize(expr: str) -> list[str]:
    expr = expr.lower().replace(",", " ")
    return [t for t in _TOKEN_RE.findall(expr) if t not in STOPWORDS and len(t) > 2]


class LexicalResources:
    def __init__(self, expressions: list[str], corpus_assoc: dict | None = None):
        self.expressions = expressions
        # associazioni da corpus: clue -> {neighbor: pmi}  (vedi mine_corpus.py)
        self.corpus_assoc: dict[str, dict[str, float]] = corpus_assoc or {}
        # indice: parola -> Counter(parole co-occorrenti)
        self.cooc: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        # per il bonus: insieme delle espressioni che contengono una data parola
        self.word_to_exprs: dict[str, set[int]] = collections.defaultdict(set)
        for idx, expr in enumerate(expressions):
            toks = set(_tokenize(expr))
            for w in toks:
                self.word_to_exprs[w].add(idx)
                for w2 in toks:
                    if w2 != w:
                        self.cooc[w][w2] += 1

    # ---- costruzione da file ----
    @classmethod
    def from_files(
        cls,
        polirematiche_path: str | Path,
        demauro_path: str | Path,
        proverbi_path: str | Path,
        wiki_titles_path: str | Path | None = None,
        corpus_assoc_path: str | Path | None = None,
    ) -> "LexicalResources":
        exprs: list[str] = []

        # polirematiche: formato "POS<TAB>espressione"
        for line in _read_lines(polirematiche_path):
            exprs.append(line.split("\t", 1)[-1])

        # demauro.poli: una espressione per riga (scarta righe senza lettere)
        for line in _read_lines(demauro_path):
            if _HAS_LETTER.search(line):
                exprs.append(line)

        # proverbi: una frase per riga
        exprs.extend(_read_lines(proverbi_path))

        # titoli Wikipedia (risorsa "Wiki-IT-Titles", come in Sangati et al.):
        # ogni titolo multi-parola è trattato come un'espressione che collega le sue parole
        if wiki_titles_path and Path(wiki_titles_path).exists():
            exprs.extend(_load_wiki_titles(wiki_titles_path))

        corpus_assoc = None
        if corpus_assoc_path and Path(corpus_assoc_path).exists():
            corpus_assoc = _load_corpus_assoc(corpus_assoc_path)

        return cls(exprs, corpus_assoc=corpus_assoc)

    # ---- API usata dal solver ----
    def generate_candidates(self, hints: list[str]) -> collections.Counter:
        """Restituisce Counter: candidato -> numero di indizi distinti con cui co-occorre."""
        scores: collections.Counter = collections.Counter()
        for h in hints:
            h = h.lower()
            seen_for_hint = set(self.cooc.get(h, {}).keys())
            for w in seen_for_hint:
                scores[w] += 1
        # un candidato non può essere uno degli indizi stessi
        for h in hints:
            scores.pop(h.lower(), None)
        return scores

    def mwe_bonus(self, candidate: str, hints: list[str]) -> int:
        """Numero di indizi che formano un'espressione nota insieme al candidato."""
        cand = candidate.lower()
        cand_exprs = self.word_to_exprs.get(cand, set())
        if not cand_exprs:
            return 0
        return sum(
            1 for h in hints if cand_exprs & self.word_to_exprs.get(h.lower(), set())
        )


def _read_lines(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _load_corpus_assoc(path: str | Path) -> dict[str, dict[str, float]]:
    """Carica il file TSV prodotto da mine_corpus.py: clue, neighbor, count, pmi."""
    out: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            clue, nb, _cnt, pmi = parts[0], parts[1], parts[2], parts[3]
            try:
                out.setdefault(clue, {})[nb] = float(pmi)
            except ValueError:
                continue
    return out


_TITLE_TOKEN = re.compile(r"^[a-zàèéìòùA-ZÀÈÉÌÒÙ]+$")


def _load_wiki_titles(path: str | Path, min_tok: int = 2, max_tok: int = 4) -> list[str]:
    """Carica i titoli di Wikipedia (dump 'all-titles-in-ns0', underscore=spazio).

    Tiene solo i titoli multi-parola composti da parole alfabetiche (2-4 token),
    scartando voci con cifre/punteggiatura/parentesi: sono associazioni rumorose.
    """
    out: list[str] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        next(f, None)  # salta l'eventuale header del dump
        for line in f:
            title = line.strip().replace("_", " ")
            toks = title.split()
            if min_tok <= len(toks) <= max_tok and all(_TITLE_TOKEN.match(t) for t in toks):
                out.append(title)
    return out
