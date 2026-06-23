"""Tuning dei parametri di scoring con embeddings.

Carica risorse ed embeddings UNA volta, poi prova diverse combinazioni di
soglia di copertura embedding (emb_threshold) e peso beta, riportando Acc@1 e MRR
sul test set. Serve a trovare i parametri che massimizzano la precisione top-1
sfruttando gli embeddings solo dove aiutano.

Uso:
    IT_EMBEDDINGS_LIMIT=300000 IT_EMBEDDINGS_PATH=~/Downloads/cc.it.300.vec \
        python tune_scoring.py
"""
from __future__ import annotations

import config
from data_utils import load_games
from resources import LexicalResources
from embeddings import EmbeddingModel
from baseline import ClassicSolver


def evaluate(solver: ClassicSolver, test) -> tuple[float, float, int]:
    acc1 = rr = 0
    topk = 0
    for g in test:
        sol = g.solution.lower()
        full = [w for w, _ in solver.solve(g.hints, top_k=10**9).ranked]
        if full and full[0] == sol:
            acc1 += 1
        if sol in full:
            pos = full.index(sol)
            if pos < 50:
                topk += 1
            rr += 1.0 / (pos + 1)
    n = len(test)
    return 100 * acc1 / n, rr / n, topk


def main() -> None:
    res = LexicalResources.from_files(
        config.POLIREMATICHE_PATH, config.DEMAURO_PATH, config.PROVERBI_PATH,
        wiki_titles_path=config.WIKI_TITLES_PATH,
    )
    emb = EmbeddingModel.load(config.EMBEDDINGS_PATH)
    if emb is None:
        print("Embeddings non trovati: imposta IT_EMBEDDINGS_PATH.")
        return

    # DEV = 200 partite del training (per la selezione dei parametri),
    # TEST = 100 partite ufficiali (solo per la valutazione finale).
    train = load_games(config.TRAIN_PATH)
    dev = train[:200]
    test = load_games(config.TEST_PATH)

    print("Selezione parametri sul DEV (200 partite di training):")
    print(f"{'thr':>5} {'beta':>5} {'nbr':>4} | {'Acc@1':>6} {'MRR':>6}")
    best = (-1.0, None)
    for thr in (0.60, 2.0):           # 2.0 = copertura solo da MWE
        for beta in (1.0, 2.0, 3.0, 4.0):
            for nbr in (30, 50):
                solver = ClassicSolver(
                    res, embeddings=emb, beta=beta,
                    emb_threshold=thr, emb_neighbors=nbr,
                )
                a, m, _ = evaluate(solver, dev)
                flag = ""
                if a > best[0]:
                    best = (a, (thr, beta, nbr)); flag = "  <-- best"
                print(f"{thr:>5} {beta:>5} {nbr:>4} | {a:>5.1f}% {m:>6.3f}{flag}")

    thr, beta, nbr = best[1]
    print(f"\nMigliori parametri sul DEV: thr={thr}, beta={beta}, neighbors={nbr} "
          f"(Acc@1 dev={best[0]:.1f}%)")

    print("\nValutazione finale sul TEST (100 partite) con i parametri scelti:")
    for label, e in (("senza embeddings", None), ("con embeddings", emb)):
        s = (ClassicSolver(res, embeddings=None) if e is None
             else ClassicSolver(res, embeddings=e, beta=beta,
                                 emb_threshold=thr, emb_neighbors=nbr))
        a, m, t = evaluate(s, test)
        print(f"  {label:18s}: Acc@1={a:.0f}%  MRR={m:.3f}  top50={t}")


if __name__ == "__main__":
    main()
