"""Pipeline completa: [A]+[B] candidati classici -> [C] scelta LLM -> [D] descrizione LLM."""
from __future__ import annotations

import time
from dataclasses import dataclass

import config
import prompts
from baseline import ClassicSolver
from data_utils import Game
from llm import LLMEngine


@dataclass
class FullPrediction:
    solution: str
    description: str
    solve_time_s: float   # tempo per TROVARE la soluzione (conta per il limite 60s)
    desc_time_s: float    # tempo per generare la descrizione (non soggetto al limite)
    candidates: list[str]
    llm_ok: bool = True   # False se la chiamata LLM è fallita e si è usata la baseline

    @property
    def elapsed_s(self) -> float:
        return self.solve_time_s + self.desc_time_s


class GhigliottinaPipeline:
    def __init__(
        self,
        solver: ClassicSolver,
        engine: LLMEngine,
        few_shot: list[Game] | None = None,
        top_k: int = config.TOP_K_CANDIDATES,
    ):
        self.solver = solver
        self.engine = engine
        self.few_shot = few_shot or []
        self.top_k = top_k

    def solve(
        self, hints: list[str], generate_desc: bool = True, llm_select: bool = False
    ) -> FullPrediction:
        # ---- fase cronometrata: TROVARE la soluzione (limite 60s) ----
        t0 = time.time()
        candidates = self.solver.candidate_pool(hints, top_k=self.top_k)  # [A]+[B]
        baseline_best = candidates[0] if candidates else ""
        llm_ok = True
        if not llm_select:
            # selezione classica: il candidato migliore del ranking knowledge-based
            solution = baseline_best
        else:
            try:
                raw = self.engine.complete(
                    prompts.build_solve_prompt(hints, candidates),
                    system=prompts.SOLVE_SYSTEM,
                    temperature=0.0,
                    max_tokens=1024,
                )
                if not raw.strip():
                    raise RuntimeError("risposta LLM vuota")
                solution = prompts.parse_choice(raw, candidates, fallback=baseline_best)
            except Exception as e:  # noqa: BLE001
                llm_ok = False  # LLM non disponibile: si usa la baseline
                self._last_error = str(e)
                solution = baseline_best
        solve_time = time.time() - t0

        # ---- fase NON cronometrata: descrizione ----
        description = ""
        t1 = time.time()
        if generate_desc and solution:
            try:
                description = self.engine.complete(
                    prompts.build_desc_prompt(hints, solution, self.few_shot),
                    system=prompts.DESC_SYSTEM,
                    temperature=0.3,
                    max_tokens=1024,
                ).strip()
            except Exception:
                description = ""
        desc_time = time.time() - t1

        return FullPrediction(
            solution=solution,
            description=description,
            solve_time_s=solve_time,
            desc_time_s=desc_time,
            candidates=candidates,
            llm_ok=llm_ok,
        )
