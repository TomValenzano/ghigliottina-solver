"""Valuta il GraphSolver (spreading activation). Tuning sul dev, report sul test.

Uso:
    IT_EMBEDDINGS_LIMIT=300000 IT_EMBEDDINGS_PATH=~/Downloads/cc.it.300.vec \
        python run_graph.py
"""
from __future__ import annotations

import time

import config
from data_utils import load_games
from resources import LexicalResources
from embeddings import EmbeddingModel
from graph_solver import GraphSolver


def evaluate(solver, games):
    acc1 = rr = 0
    for g in games:
        sol = g.solution.lower()
        ranked = [w for w, _ in solver.solve(g.hints, top_k=10**9).ranked]
        if ranked and ranked[0] == sol:
            acc1 += 1
        if sol in ranked:
            rr += 1.0 / (ranked.index(sol) + 1)
    n = len(games)
    return 100 * acc1 / n, rr / n


def main() -> None:
    res = LexicalResources.from_files(
        config.POLIREMATICHE_PATH, config.DEMAURO_PATH, config.PROVERBI_PATH
    )
    emb = EmbeddingModel.load(config.EMBEDDINGS_PATH)
    if emb is None:
        print("Embeddings non trovati: imposta IT_EMBEDDINGS_PATH.")
        return
    train = load_games(config.TRAIN_PATH)
    dev = train[:150]
    test = load_games(config.TEST_PATH)

    print("Tuning GraphSolver sul DEV (150 partite):")
    print(f"{'tau':>5} {'embW':>5} {'rest':>5} {'nbr':>4} | {'Acc@1':>6} {'MRR':>6} {'t/g':>6}")
    best = (-1.0, None)
    grid = [
        (0.30, 1.0, 0.40, 50),
        (0.25, 1.0, 0.40, 50),
        (0.35, 0.5, 0.40, 50),
        (0.30, 0.5, 0.30, 60),
    ]
    for tau, embW, rest, nbr in grid:
        gs = GraphSolver(res, emb, emb_neighbors=nbr, tau=tau,
                         emb_weight=embW, restart=rest)
        t0 = time.time()
        a, m = evaluate(gs, dev)
        tg = (time.time() - t0) / len(dev)
        flag = ""
        if a > best[0]:
            best = (a, (tau, embW, rest, nbr)); flag = "  <-- best"
        print(f"{tau:>5} {embW:>5} {rest:>5} {nbr:>4} | {a:>5.1f}% {m:>6.3f} {tg:>5.2f}s{flag}")

    tau, embW, rest, nbr = best[1]
    print(f"\nMigliore sul DEV: tau={tau}, embW={embW}, restart={rest}, nbr={nbr} "
          f"(Acc@1={best[0]:.1f}%)")
    gs = GraphSolver(res, emb, emb_neighbors=nbr, tau=tau, emb_weight=embW, restart=rest)
    a, m = evaluate(gs, test)
    print(f"\nGraphSolver sul TEST: Acc@1={a:.0f}%  MRR={m:.3f}")
    print("(riferimento: solver classico = 31% Acc@1, MRR 0.37)")


if __name__ == "__main__":
    main()
