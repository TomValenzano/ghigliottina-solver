"""Confronta lo scoring 'coverage' (rigido) vs 'assoc' (PMI, soft) sul dev e sul test.

Pensato per la knowledge base arricchita coi titoli Wikipedia (recall alto): verifica
se la pesatura per specificità sblocca i gold sepolti in classifica.

Uso:
    WIKI_TITLES_PATH=~/Downloads/itwiki-latest-all-titles-in-ns0 python run_compare_scoring.py
    (aggiungi IT_EMBEDDINGS_PATH per includere anche gli embeddings)
"""
from __future__ import annotations

import config
from data_utils import load_games
from resources import LexicalResources
from embeddings import EmbeddingModel
from baseline import ClassicSolver


def evaluate(solver, games):
    acc1 = rr = top50 = 0
    for g in games:
        sol = g.solution.lower()
        ranked = [w for w, _ in solver.solve(g.hints, top_k=10**9).ranked]
        if ranked and ranked[0] == sol:
            acc1 += 1
        if sol in ranked:
            pos = ranked.index(sol)
            if pos < 50:
                top50 += 1
            rr += 1.0 / (pos + 1)
    n = len(games)
    return 100 * acc1 / n, rr / n, 100 * top50 / n


def main() -> None:
    res = LexicalResources.from_files(
        config.POLIREMATICHE_PATH, config.DEMAURO_PATH, config.PROVERBI_PATH,
        wiki_titles_path=config.WIKI_TITLES_PATH,
        corpus_assoc_path=config.CORPUS_ASSOC_PATH,
    )
    emb = EmbeddingModel.load(config.EMBEDDINGS_PATH)
    print(f"{len(res.expressions)} espressioni | embeddings: {'sì' if emb else 'no'}")
    train = load_games(config.TRAIN_PATH)
    dev, test = train[:200], load_games(config.TEST_PATH)

    for mode in ("coverage", "assoc"):
        # per 'assoc' proviamo un paio di pesi beta sul dev se ci sono embeddings
        betas = ([0.0, 1.0] if emb else [0.0])
        best = (-1, None)
        for beta in betas:
            s = ClassicSolver(res, embeddings=emb, beta=beta,
                              emb_threshold=config.EMB_THRESHOLD,
                              emb_neighbors=config.EMB_NEIGHBORS, scoring=mode)
            a, m, t = evaluate(s, dev)
            if a > best[0]:
                best = (a, beta)
        beta = best[1]
        s = ClassicSolver(res, embeddings=emb, beta=beta,
                          emb_threshold=config.EMB_THRESHOLD,
                          emb_neighbors=config.EMB_NEIGHBORS, scoring=mode)
        da, dm, dt = evaluate(s, dev)
        ta, tm, tt = evaluate(s, test)
        print(f"\n[{mode}] (beta={beta})")
        print(f"  DEV : Acc@1={da:.1f}%  MRR={dm:.3f}  top50={dt:.0f}%")
        print(f"  TEST: Acc@1={ta:.0f}%  MRR={tm:.3f}  top50={tt:.0f}%")


if __name__ == "__main__":
    main()
