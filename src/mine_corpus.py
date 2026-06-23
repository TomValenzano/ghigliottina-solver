"""Corpus mining mirato: collocati delle parole-indizio (stile Sangati, semplificato).

Fa UNA passata in streaming su un corpus di testo italiano e, per ogni occorrenza di
una parola-indizio (tutte note: train+test), conta le parole-contenuto vicine entro
una finestra. Calcola poi un punteggio PMI per ogni coppia (indizio, vicino) e salva
le associazioni in un file TSV, da usare come fonte aggiuntiva nel solver.

Memoria limitata: si tracciano solo i collocati degli indizi (~1500 parole), con
potatura periodica dei conteggi bassi.

Uso:
    python mine_corpus.py --corpus ~/Downloads/paisa.raw.utf8 --out ../data/corpus_assoc.tsv
    # opzioni: --window 3 --min-count 5 --top 300
"""
from __future__ import annotations

import argparse
import collections
import math
import re

import config
from data_utils import load_games
from resources import STOPWORDS

TOKEN_RE = re.compile(r"[a-zàèéìòù]+")


def content_tokens(line: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(line.lower()) if len(t) > 2 and t not in STOPWORDS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="file di testo (una riga = una frase o paragrafo)")
    ap.add_argument("--out", default=str(config.DATA_DIR.parent / "corpus_assoc.tsv"))
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--top", type=int, default=300, help="max vicini per indizio nel file finale")
    ap.add_argument("--prune-every", type=int, default=20_000_000, help="token tra una potatura e l'altra")
    args = ap.parse_args()

    # parole-indizio target (train + test)
    clues: set[str] = set()
    for path in (config.TRAIN_PATH, config.TEST_PATH):
        for g in load_games(path):
            clues.update(h.lower() for h in g.hints)
    print(f"{len(clues)} parole-indizio target")

    cooc: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    freq: collections.Counter = collections.Counter()
    total = 0
    W = args.window

    def prune():
        for k in list(cooc.keys()):
            c = cooc[k]
            if len(c) > args.top * 4:
                cooc[k] = collections.Counter(dict(c.most_common(args.top * 2)))

    with open(args.corpus, encoding="utf-8", errors="ignore") as f:
        for line in f:
            toks = content_tokens(line)
            for t in toks:
                freq[t] += 1
            total += len(toks)
            for i, t in enumerate(toks):
                if t in clues:
                    lo, hi = max(0, i - W), min(len(toks), i + W + 1)
                    for j in range(lo, hi):
                        if j != i:
                            cooc[t][toks[j]] += 1
            if total >= args.prune_every:
                prune()
                total_seen = sum(freq.values())
                print(f"  ...{total_seen:,} token processati")
                total = 0

    # PMI e scrittura
    N = max(1, sum(freq.values()))
    n_pairs = 0
    with open(args.out, "w", encoding="utf-8") as out:
        out.write("clue\tneighbor\tcount\tpmi\n")
        for clue, neighbors in cooc.items():
            fc = freq.get(clue, 0)
            if fc == 0:
                continue
            scored = []
            for nb, cnt in neighbors.items():
                if cnt < args.min_count or nb == clue:
                    continue
                fn = freq.get(nb, 0)
                if fn == 0:
                    continue
                pmi = math.log((cnt * N) / (fc * fn) + 1e-12)
                if pmi > 0:  # tieni solo associazioni positive
                    scored.append((nb, cnt, pmi))
            scored.sort(key=lambda x: x[2], reverse=True)
            for nb, cnt, pmi in scored[: args.top]:
                out.write(f"{clue}\t{nb}\t{cnt}\t{pmi:.4f}\n")
                n_pairs += 1
    print(f"Scritte {n_pairs} associazioni in {args.out}")


if __name__ == "__main__":
    main()
