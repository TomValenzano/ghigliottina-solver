"""Esegue la pipeline completa sul test set e salva le predizioni in JSON.

Uso:
    GEMINI_API_KEY=... python run_full.py --engine gemini
    python run_full.py --engine ollama
    python run_full.py --engine mock            # test offline
    python run_full.py --engine gemini --limit 10   # solo prime 10 partite
"""
from __future__ import annotations

import argparse
import json

import config
import prompts
from baseline import ClassicSolver
from data_utils import load_games
from embeddings import EmbeddingModel
from llm import get_engine
from pipeline import GhigliottinaPipeline
from resources import LexicalResources


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="mock", choices=["gemini", "ollama", "mock"])
    ap.add_argument("--limit", type=int, default=0, help="0 = tutte le partite")
    ap.add_argument("--no-desc", action="store_true", help="salta la descrizione")
    ap.add_argument("--pace", type=float, default=5.0,
                    help="secondi di pausa tra le partite (evita i rate limit; 0=off)")
    ap.add_argument("--select", choices=["classic", "llm"], default="classic",
                    help="selezione soluzione: 'classic' (knowledge-based) o 'llm'")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "predictions.json"))
    args = ap.parse_args()

    res = LexicalResources.from_files(
        config.POLIREMATICHE_PATH, config.DEMAURO_PATH, config.PROVERBI_PATH,
        wiki_titles_path=config.WIKI_TITLES_PATH,
        corpus_assoc_path=config.CORPUS_ASSOC_PATH,
    )
    emb = EmbeddingModel.load(config.EMBEDDINGS_PATH)
    solver = ClassicSolver(
        res, embeddings=emb, beta=config.BETA_EMB,
        emb_threshold=config.EMB_THRESHOLD, emb_neighbors=config.EMB_NEIGHBORS,
        corpus_pmi_threshold=config.CORPUS_PMI_THRESHOLD, gamma=config.CORPUS_GAMMA,
    )

    train = load_games(config.TRAIN_PATH)
    test = load_games(config.TEST_PATH)
    if args.limit:
        test = test[: args.limit]

    engine = get_engine(args.engine)
    few_shot = prompts.sample_few_shot(train, k=2)
    pipe = GhigliottinaPipeline(solver, engine, few_shot=few_shot)

    import time

    out = []
    correct = 0
    fallbacks = 0
    for i, g in enumerate(test, 1):
        # pacing proattivo tra le partite (fuori dal cronometro) per non
        # superare i rate limit del free tier; disattivabile con --pace 0
        if i > 1 and args.pace > 0:
            time.sleep(args.pace)

        pred = pipe.solve(g.hints, generate_desc=not args.no_desc,
                          llm_select=(args.select == "llm"))
        ok = pred.solution == g.solution.lower()
        # il limite 60s vale solo sul tempo di RISOLUZIONE, non sulla descrizione
        over = pred.solve_time_s > config.TIME_LIMIT_S
        correct += ok
        if not pred.llm_ok:
            fallbacks += 1
        out.append({
            "id": g.id,
            "hints": g.hints,
            "gold_solution": g.solution,
            "pred_solution": pred.solution,
            "correct": ok,
            "llm_ok": pred.llm_ok,
            "gold_desc": g.description,
            "pred_desc": pred.description,
            "solve_time_s": round(pred.solve_time_s, 3),
            "desc_time_s": round(pred.desc_time_s, 3),
            "over_60s": over,
        })
        print(f"[{i}/{len(test)}] {g.hints} -> {pred.solution} "
              f"(gold {g.solution}) {'OK' if ok else 'X'} "
              f"{'' if pred.llm_ok else '[BASELINE-FALLBACK!] '}"
              f"solve {pred.solve_time_s:.1f}s | desc {pred.desc_time_s:.1f}s"
              f"{'  >60s!' if over else ''}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nAccuracy soluzione: {correct}/{len(test)} ({100*correct/len(test):.1f}%)")
    if fallbacks:
        print(f"ATTENZIONE: {fallbacks}/{len(test)} partite hanno usato la BASELINE "
              f"(LLM non disponibile). I numeri NON riflettono l'LLM!")
        if getattr(pipe, "_last_error", ""):
            print(f"Ultimo errore LLM: {pipe._last_error[:200]}")
    print(f"Predizioni salvate in {args.out}")


if __name__ == "__main__":
    main()
