"""Valuta la baseline classica sul test set.

Uso:
    python run_baseline.py
    IT_EMBEDDINGS_PATH=/percorso/cc.it.300.bin python run_baseline.py

Se IT_EMBEDDINGS_PATH è impostato e il file esiste, lo scoring usa anche gli
embeddings (ri-ordinamento dei candidati); altrimenti usa il solo segnale
delle polirematiche.
"""
from __future__ import annotations

import time

import config
from data_utils import load_games
from resources import LexicalResources
from embeddings import EmbeddingModel
from baseline import ClassicSolver


def main() -> None:
    res = LexicalResources.from_files(
        config.POLIREMATICHE_PATH, config.DEMAURO_PATH, config.PROVERBI_PATH,
        wiki_titles_path=config.WIKI_TITLES_PATH,
        corpus_assoc_path=config.CORPUS_ASSOC_PATH,
    )
    emb = EmbeddingModel.load(config.EMBEDDINGS_PATH)
    mode = "polirematiche + embeddings" if emb else "solo polirematiche"
    print(f"Modalità baseline: {mode} | {len(res.expressions)} espressioni")

    solver = ClassicSolver(
        res, embeddings=emb, beta=config.BETA_EMB,
        emb_threshold=config.EMB_THRESHOLD, emb_neighbors=config.EMB_NEIGHBORS,
        corpus_pmi_threshold=config.CORPUS_PMI_THRESHOLD, gamma=config.CORPUS_GAMMA,
    )
    test = load_games(config.TEST_PATH)

    top1 = topk = recall = 0
    over_limit = 0
    times: list[float] = []
    K = config.TOP_K_CANDIDATES
    for g in test:
        t = time.time()
        pred = solver.solve(g.hints, top_k=K)
        dt = time.time() - t
        times.append(dt)
        if dt > config.TIME_LIMIT_S:
            over_limit += 1
        sol = g.solution.lower()
        ranked = [w for w, _ in pred.ranked]
        top1 += pred.solution == sol
        topk += sol in ranked
        # recall sull'INTERO insieme candidati (MWE + espansione embeddings)
        full_pool = {w for w, _ in solver.solve(g.hints, top_k=10**9).ranked}
        recall += sol in full_pool

    n = len(test)
    print(f"\n=== Risultati baseline ({mode}) ===")
    print(f"Accuracy top-1     : {top1}/{n} ({100*top1/n:.1f}%)")
    print(f"Soluzione nei top-{K}: {topk}/{n} ({100*topk/n:.1f}%)")
    print(f"Recall candidati   : {recall}/{n} ({100*recall/n:.1f}%)")
    print(f"Tempo medio/partita: {1000*sum(times)/n:.1f} ms")
    print(f"Partite oltre {config.TIME_LIMIT_S:.0f}s: {over_limit}")


if __name__ == "__main__":
    main()
