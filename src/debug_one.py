"""Debug: esegue UNA partita e stampa prompt, risposta grezza dell'LLM e parsing.

Uso:
    python debug_one.py --engine gemini --idx 13   # partita 14 (lana, gold in candidati)
"""
from __future__ import annotations

import argparse

import config
import prompts
from baseline import ClassicSolver
from data_utils import load_games
from embeddings import EmbeddingModel
from llm import get_engine
from resources import LexicalResources


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="gemini")
    ap.add_argument("--idx", type=int, default=13)
    args = ap.parse_args()

    res = LexicalResources.from_files(
        config.POLIREMATICHE_PATH, config.DEMAURO_PATH, config.PROVERBI_PATH
    )
    emb = EmbeddingModel.load(config.EMBEDDINGS_PATH)
    solver = ClassicSolver(res, embeddings=emb, alpha=config.ALPHA_MWE, beta=config.BETA_EMB)
    g = load_games(config.TEST_PATH)[args.idx]

    candidates = solver.candidate_pool(g.hints, top_k=config.TOP_K_CANDIDATES)
    prompt = prompts.build_solve_prompt(g.hints, candidates)

    print("INDIZI :", g.hints)
    print("GOLD   :", g.solution)
    print("GOLD nei candidati:", g.solution.lower() in candidates,
          f"(posizione {candidates.index(g.solution.lower())+1})"
          if g.solution.lower() in candidates else "")
    print("\n===== SYSTEM =====\n", prompts.SOLVE_SYSTEM)
    print("\n===== PROMPT =====\n", prompt)

    engine = get_engine(args.engine)
    raw = engine.complete(prompt, system=prompts.SOLVE_SYSTEM, temperature=0.0, max_tokens=1024)
    print("\n===== RISPOSTA GREZZA =====\n", repr(raw))
    print("\n===== PARSED =====", prompts.parse_choice(raw, candidates))


if __name__ == "__main__":
    main()
