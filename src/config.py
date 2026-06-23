"""Percorsi e parametri di configurazione del progetto."""
from __future__ import annotations

import os
from pathlib import Path

# Radice del progetto e cartelle dati (il dataset resta nella cartella sorella)
PROJECT_DIR = Path(__file__).resolve().parent.parent
TASK_DIR = PROJECT_DIR.parent  # .../Task 1 - Ghigliottina

DATA_DIR = TASK_DIR / "dataset_ghigliottina"
LEX_DIR = TASK_DIR / "polirematiche_proverbi"
RESULTS_DIR = PROJECT_DIR / "results"

TRAIN_PATH = DATA_DIR / "train.json"
TEST_PATH = DATA_DIR / "test.json"

POLIREMATICHE_PATH = LEX_DIR / "polirematiche"
DEMAURO_PATH = LEX_DIR / "demauro.poli"
PROVERBI_PATH = LEX_DIR / "proverbi"

# Titoli Wikipedia italiani (risorsa opzionale per alzare il recall).
# Imposta WIKI_TITLES_PATH al file scaricato (itwiki-latest-all-titles-in-ns0).
WIKI_TITLES_PATH = os.environ.get("WIKI_TITLES_PATH", "")

# Associazioni minate da corpus (prodotte da mine_corpus.py)
CORPUS_ASSOC_PATH = os.environ.get("CORPUS_ASSOC_PATH", "")
CORPUS_PMI_THRESHOLD = 3.0  # PMI minimo perché il corpus "copra" un indizio (tarato sul dev)
CORPUS_GAMMA = 0.0          # peso additivo del corpus nello score (0 = solo copertura)

# Embeddings italiani: imposta il percorso del file (es. fastText cc.it.300.bin/.vec)
# oppure tramite variabile d'ambiente IT_EMBEDDINGS_PATH.
EMBEDDINGS_PATH = os.environ.get("IT_EMBEDDINGS_PATH", "")

# Parametri scoring (selezionati sul dev set: vedi tune_scoring.py)
ALPHA_MWE = 1.0       # (compatibilità; non più usato nello scoring pesato)
BETA_EMB = 3.0        # peso della similarità embedding nello scoring
EMB_THRESHOLD = 2.0   # soglia copertura via embedding (2.0 = copertura solo da MWE)
EMB_NEIGHBORS = 30    # vicini semantici per espandere il pool candidati
TOP_K_CANDIDATES = 50  # quanti candidati passare all'LLM nella fase [C]

# Vincolo di gioco
TIME_LIMIT_S = 60.0

# LLM
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
