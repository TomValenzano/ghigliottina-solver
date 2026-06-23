"""Tuning dei parametri delle associazioni da corpus (soglia PMI e peso gamma).

Tara su DEV (200 partite di training), valuta su TEST. Richiede CORPUS_ASSOC_PATH
(file prodotto da mine_corpus.py) e i titoli Wikipedia.

Uso:
    WIKI_TITLES_PATH=~/Downloads/itwiki-latest-all-titles-in-ns0 \
    CORPUS_ASSOC_PATH=../data/corpus_assoc.tsv \
        python run_corpus.py
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
    print(f"{len(res.expressions)} espressioni | corpus_assoc su {len(res.corpus_assoc)} indizi "
          f"| embeddings: {'sì' if emb else 'no'}")
    train = load_games(config.TRAIN_PATH)
    dev, test = train[:200], load_games(config.TEST_PATH)

    # riferimento: senza corpus (corpus_assoc svuotato)
    res_nocorp = LexicalResources(res.expressions)  # stesse espressioni, niente corpus
    base = ClassicSolver(res_nocorp, embeddings=emb, beta=config.BETA_EMB,
                         emb_threshold=config.EMB_THRESHOLD, emb_neighbors=config.EMB_NEIGHBORS)
    a, m, t = evaluate(base, test)
    print(f"\nRiferimento (titoli, NO corpus) sul TEST: Acc@1={a:.0f}%  MRR={m:.3f}  top50={t:.0f}%")

    print("\nTuning corpus sul DEV:")
    print(f"{'pmiThr':>6} {'gamma':>6} | {'Acc@1':>6} {'MRR':>6} {'top50':>6}")
    best = (-1.0, None)
    for thr in (1.5, 2.0, 3.0, 4.0):
        for gamma in (0.0, 0.3, 1.0):
            s = ClassicSolver(res, embeddings=emb, beta=config.BETA_EMB,
                              emb_threshold=config.EMB_THRESHOLD,
                              emb_neighbors=config.EMB_NEIGHBORS,
                              corpus_pmi_threshold=thr, gamma=gamma)
            a, m, t = evaluate(s, dev)
            flag = ""
            if a > best[0]:
                best = (a, (thr, gamma)); flag = "  <-- best"
            print(f"{thr:>6} {gamma:>6} | {a:>5.1f}% {m:>6.3f} {t:>5.0f}%{flag}")

    thr, gamma = best[1]
    s = ClassicSolver(res, embeddings=emb, beta=config.BETA_EMB,
                      emb_threshold=config.EMB_THRESHOLD, emb_neighbors=config.EMB_NEIGHBORS,
                      corpus_pmi_threshold=thr, gamma=gamma)
    a, m, t = evaluate(s, test)
    print(f"\nMigliore sul DEV: pmiThr={thr}, gamma={gamma} (dev Acc@1={best[0]:.1f}%)")
    print(f"CON corpus sul TEST: Acc@1={a:.0f}%  MRR={m:.3f}  top50={t:.0f}%")


if __name__ == "__main__":
    main()
