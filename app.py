"""Demo interattiva del solver (Streamlit).

Avvio:
    pip install streamlit
    WIKI_TITLES_PATH=/path/itwiki-latest-all-titles-in-ns0 \
    CORPUS_ASSOC_PATH=data/corpus_assoc.tsv \
        streamlit run app.py

La knowledge base viene caricata una sola volta (cache); la risoluzione è
istantanea. La descrizione richiede Ollama attivo (`ollama serve`) ed è
opzionale.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

import config
import prompts
from baseline import ClassicSolver
from data_utils import load_games
from resources import LexicalResources

TOP_N = 10
# Partite reali del test set risolte correttamente dal sistema
EXAMPLES = [
    ("cassa", ["legno", "forte", "continua", "malattia", "veloce"]),
    ("sonno", ["ore", "pesante", "cura", "arretrato", "cascare"]),
]


# ------------------------------------------------------------------ caching
@st.cache_resource(show_spinner="Carico la knowledge base (una tantum)...")
def load_resources() -> LexicalResources:
    return LexicalResources.from_files(
        config.POLIREMATICHE_PATH, config.DEMAURO_PATH, config.PROVERBI_PATH,
        wiki_titles_path=config.WIKI_TITLES_PATH,
        corpus_assoc_path=config.CORPUS_ASSOC_PATH,
    )


@st.cache_resource(show_spinner=False)
def load_few_shot():
    try:
        return prompts.sample_few_shot(load_games(config.TRAIN_PATH), k=2)
    except Exception:  # noqa: BLE001 — demo utilizzabile anche senza train
        return []


# ------------------------------------------------------------------ helpers
def clue_evidence(res: LexicalResources, cand: str, hints: list[str]):
    """Per ogni indizio: espressioni condivise con il candidato e PMI da corpus."""
    out = []
    cand_exprs = res.word_to_exprs.get(cand, set())
    for h in hints:
        h = h.lower()
        shared = cand_exprs & res.word_to_exprs.get(h, set())
        exprs = [res.expressions[i] for i in sorted(shared)[:3]]
        pmi = res.corpus_assoc.get(h, {}).get(cand)
        out.append((h, exprs, pmi))
    return out


def coverage_row(res: LexicalResources, cand: str, hints: list[str]) -> list[bool]:
    row = []
    cand_exprs = res.word_to_exprs.get(cand, set())
    for h in hints:
        h = h.lower()
        mwe = bool(cand_exprs & res.word_to_exprs.get(h, set()))
        pmi = res.corpus_assoc.get(h, {}).get(cand, 0.0)
        row.append(mwe or pmi >= config.CORPUS_PMI_THRESHOLD)
    return row


# ------------------------------------------------------------------ UI
st.set_page_config(page_title="La Ghigliottina — Solver", page_icon="🔪", layout="wide")
st.title("La Ghigliottina — solver knowledge-based")
st.caption(
    "Cinque indizi, una parola nascosta. Il ranker knowledge-based "
    "(MWE + titoli Wikipedia + collocazioni da corpus) la trova in ~0.01 s; "
    "l'LLM locale serve solo a spiegarla."
)

res = load_resources()
solver = ClassicSolver(
    res, embeddings=None, beta=config.BETA_EMB,
    emb_threshold=config.EMB_THRESHOLD, emb_neighbors=config.EMB_NEIGHBORS,
    corpus_pmi_threshold=config.CORPUS_PMI_THRESHOLD, gamma=config.CORPUS_GAMMA,
)

with st.sidebar:
    st.header("Knowledge base")
    st.write(f"**{len(res.expressions):,}** espressioni indicizzate")
    st.write(f"**{len(res.corpus_assoc):,}** indizi con collocazioni da corpus")
    if not config.WIKI_TITLES_PATH:
        st.warning("WIKI_TITLES_PATH non impostato: titoli Wikipedia esclusi.")
    if not config.CORPUS_ASSOC_PATH:
        st.warning("CORPUS_ASSOC_PATH non impostato: collocazioni escluse.")
    st.write(f"Soglia PMI copertura: **{config.CORPUS_PMI_THRESHOLD}**")
    st.divider()
    st.caption("Esempi dal test set:")
    for sol, words in EXAMPLES:
        if st.button(f"Soluzione: {sol}", key=f"ex_{sol}"):
            for i, w in enumerate(words):
                st.session_state[f"hint{i}"] = w

cols = st.columns(5)
hints = [
    c.text_input(f"Indizio {i + 1}", key=f"hint{i}").strip()
    for i, c in enumerate(cols)
]

if all(hints):
    t0 = time.perf_counter()
    pred = solver.solve(hints, top_k=TOP_N)
    dt = time.perf_counter() - t0

    if not pred.solution:
        st.error("Nessun candidato trovato per questi indizi.")
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("Soluzione", pred.solution.upper())
    cov = sum(coverage_row(res, pred.solution, hints))
    m2.metric("Copertura indizi", f"{cov}/5")
    m3.metric("Tempo di risoluzione", f"{dt * 1000:.0f} ms")

    # ---- classifica ----
    st.subheader(f"Top {TOP_N} candidati")
    rows = []
    for rank, (w, score) in enumerate(pred.ranked, 1):
        flags = coverage_row(res, w, hints)
        row = {"#": rank, "candidato": w, "score": round(score, 1)}
        row.update({h: ("✓" if f else "–") for h, f in zip(hints, flags)})
        rows.append(row)
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # ---- evidenze ----
    st.subheader("Perché? Le associazioni nella knowledge base")
    chosen = st.selectbox(
        "Candidato da spiegare", [w for w, _ in pred.ranked], index=0
    )
    for h, exprs, pmi in clue_evidence(res, chosen, hints):
        bits = []
        if exprs:
            bits.append(" · ".join(f"“{e}”" for e in exprs))
        if pmi is not None:
            bits.append(f"collocazione da corpus (PMI {pmi:.1f})")
        icon = "✓" if bits else "✗"
        st.markdown(f"{icon} **{h}** — {'; '.join(bits) if bits else 'nessuna associazione'}")

    # ---- descrizione (opzionale, LLM locale) ----
    st.subheader("Descrizione (LLM locale, fuori dal limite dei 60 s)")
    if st.button(f"Genera con {config.OLLAMA_MODEL} via Ollama"):
        from llm import OllamaEngine

        try:
            with st.spinner("Generazione in corso (~7 s)..."):
                t1 = time.perf_counter()
                desc = OllamaEngine().complete(
                    prompts.build_desc_prompt(hints, pred.solution, load_few_shot()),
                    system=prompts.DESC_SYSTEM,
                    temperature=0.3,
                    max_tokens=1024,
                ).strip()
            st.success(desc)
            st.caption(f"Generata in {time.perf_counter() - t1:.1f} s")
        except Exception as e:  # noqa: BLE001
            st.error(
                f"Ollama non raggiungibile ({e}). Avvia `ollama serve` e "
                f"assicurati di avere il modello: `ollama pull {config.OLLAMA_MODEL}`."
            )
else:
    st.info("Inserisci i cinque indizi (o usa l'esempio nella barra laterale).")
