"""Costruzione dei prompt per la fase [C] (scelta soluzione) e [D] (descrizione)."""
from __future__ import annotations

import json
import random

from data_utils import Game

# ---------------------------------------------------------------- [C] scelta
N_CHOICES = 25  # quanti candidati mostrare all'LLM

SOLVE_SYSTEM = (
    "Sei un esperto del gioco televisivo 'La Ghigliottina'. Ricevi cinque parole "
    "indizio: esiste UNA sola parola italiana collegata a TUTTE e cinque. Il legame "
    "è tipicamente un'espressione, un modo di dire, una parola composta o "
    "un'associazione comune con ciascun indizio (es. indizi 'doppio, carta, soldi, "
    "pasta, regalo' -> 'pacco': doppio pacco, carta da pacco, pacco di soldi, pacco "
    "di pasta, pacco regalo).\n"
    "Ti viene data una lista NUMERATA di parole candidate, ordinate per forza "
    "dell'associazione. Devi scegliere QUALE candidato della lista è la soluzione: "
    "la parola che si collega meglio a tutti e cinque gli indizi. Considera prima "
    "i candidati in alto. NON inventare parole nuove e NON scegliere un indizio.\n"
    "Ragiona brevemente sul legame dei candidati più promettenti con i cinque "
    'indizi, poi termina con UNA riga finale contenente SOLO il JSON: '
    '{"choice": N} dove N è il numero del candidato scelto.'
)


def build_solve_prompt(hints: list[str], candidates: list[str]) -> str:
    if candidates:
        cand_block = "\n".join(
            f"  {i}. {w}" for i, w in enumerate(candidates[:N_CHOICES], 1)
        )
    else:
        cand_block = "  (nessun candidato)"
    return (
        f"Indizi: {', '.join(hints)}\n"
        f"Candidati (ordinati per forza di associazione):\n{cand_block}\n\n"
        f'Quale candidato è la soluzione? Ragiona e concludi con {{"choice": N}}.'
    )


def parse_choice(text: str, candidates: list[str], fallback: str = "") -> str:
    """Estrae il numero scelto dall'LLM e lo mappa al candidato corrispondente."""
    import re

    text = text.strip()
    # cerca l'ultimo {"choice": N}
    for m in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            obj = json.loads(m)
        except json.JSONDecodeError:
            continue
        if "choice" in obj:
            try:
                idx = int(obj["choice"]) - 1
            except (ValueError, TypeError):
                continue
            if 0 <= idx < len(candidates):
                return candidates[idx]
    # fallback: ultimo numero presente nel testo
    nums = re.findall(r"\b(\d{1,2})\b", text)
    if nums:
        idx = int(nums[-1]) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    return fallback


def parse_solution(text: str, fallback: str = "") -> str:
    """Estrae la parola-soluzione dalla risposta dell'LLM (JSON o testo)."""
    text = text.strip()
    # prova a estrarre l'ULTIMO oggetto JSON con chiave "solution"
    import re
    for m in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            obj = json.loads(m)
        except json.JSONDecodeError:
            continue
        sol = str(obj.get("solution", "")).strip().lower()
        if sol:
            return sol.split()[0]
    # fallback: prima parola-contenuto (salta articoli/preposizioni e "soluzione")
    import re
    skip = {"la", "il", "lo", "è", "e", "soluzione", "parola", "una", "un", "the"}
    for w in re.findall(r"[a-zàèéìòù]+", text.lower()):
        if w not in skip and len(w) > 1:
            return w
    return fallback


# ----------------------------------------------------------- [D] descrizione
DESC_SYSTEM = (
    "Sei un esperto del gioco 'La Ghigliottina'. Data la soluzione e i cinque "
    "indizi, scrivi una descrizione in italiano (circa 80-90 parole) che spieghi "
    "il legame tra la soluzione e CIASCUNO dei cinque indizi. Usa esattamente "
    "questo formato: inizia con la soluzione con la maiuscola seguita da due "
    "punti, poi un trattino per ogni legame. Esempio di struttura: "
    "'Soluzione: -legame con indizio1; -legame con indizio2; ...'."
)


def build_desc_prompt(
    hints: list[str], solution: str, few_shot: list[Game] | None = None
) -> str:
    parts: list[str] = []
    for ex in few_shot or []:
        parts.append(
            f"Indizi: {', '.join(ex.hints)}\nSoluzione: {ex.solution}\n"
            f"Descrizione: {ex.description}\n"
        )
    parts.append(
        f"Indizi: {', '.join(hints)}\nSoluzione: {solution}\nDescrizione:"
    )
    return "\n".join(parts)


def sample_few_shot(train: list[Game], k: int = 2, seed: int = 0) -> list[Game]:
    """Campiona k esempi del train che hanno una descrizione."""
    with_desc = [g for g in train if g.has_description]
    rng = random.Random(seed)
    return rng.sample(with_desc, min(k, len(with_desc)))
