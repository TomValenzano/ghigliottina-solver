# La Ghigliottina — Solver

Sistema per risolvere il gioco televisivo italiano *La Ghigliottina*: dati cinque
indizi, trovare l'unica parola-soluzione collegata a tutti e cinque e generarne una
descrizione. Progetto per il corso di Natural Language Processing, Università di
Bari Aldo Moro (2025-2026).

## Risultati

| Metrica | Valore |
|---|---|
| Accuracy top-1 (soluzione) | **47%** |
| MRR | 0.55 |
| Tempo medio di risoluzione | 0.01 s/partita (0 partite oltre 60 s) |
| Descrizione — BERTScore-F1 (BERT multilingue) | 0.73 |
| Descrizione — ROUGE-1 / ROUGE-L / BLEU | 0.33 / 0.20 / 6.73 |

Test set ufficiale: 100 partite. Dettagli in [`results/01_results.md`](results/01_results.md).

## Approccio

La soluzione è trovata da un **ranker knowledge-based**; un LLM locale genera solo
la descrizione.

```
5 indizi ─► [A] generazione candidati ─► [B] scoring per copertura ─► SOLUZIONE
                                                                   └─► descrizione (LLM)
```

- **Knowledge base.** Tre fonti di associazioni, indicizzate per co-occorrenza:
  polirematiche/proverbi (incl. dizionario De Mauro), titoli multi-parola di
  Wikipedia italiana, e collocazioni (PMI) estratte dal corpus Paisà.
- **Scoring.** Ogni candidato è valutato per *copertura* (a quanti dei 5 indizi è
  associato) con penalità di rarità per non favorire le parole comuni; vince chi è
  collegato a tutti e cinque gli indizi. Parametri scelti su un dev set.
- **Descrizione.** Generata da Qwen2.5-7B (locale, via Ollama), con esempi few-shot
  dalle descrizioni del training.

L'approccio si ispira a OTTHO e UNIOR4NLP / *Il Mago della Ghigliottina* (Sangati et
al.). Embeddings fastText e uno scorer a grafo (Personalized PageRank) sono stati
valutati ma non migliorano il sistema finale; allo stesso modo, usare un LLM per
*scegliere* la soluzione peggiora il ranker knowledge-based (vedi ablation in
`results/01_results.md`).

## Struttura

```
ghigliottina_solver/
├── src/
│   ├── data_utils.py        # caricamento dataset JSONL
│   ├── config.py            # percorsi e parametri
│   ├── resources.py         # knowledge base (MWE, titoli, corpus) e indice co-occorrenza
│   ├── embeddings.py        # wrapper embeddings (opzionale, non nel sistema finale)
│   ├── baseline.py          # ranker knowledge-based (scoring per copertura)
│   ├── graph_solver.py      # scorer a grafo (ablation)
│   ├── llm.py / prompts.py  # motore LLM e prompt per la descrizione
│   ├── pipeline.py          # pipeline soluzione + descrizione
│   ├── mine_corpus.py       # estrazione collocazioni da corpus
│   ├── run_baseline.py / run_full.py / run_eval.py
│   └── run_corpus.py / tune_scoring.py / run_graph.py / run_compare_scoring.py
├── results/                 # metriche e analisi
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Dati (non inclusi nel repository)

Vanno collocati nelle cartelle indicate e scaricati separatamente:

- **Dataset** del task (`train.json`, `test.json`) e **liste lessicali**
  (`polirematiche`, `demauro.poli`, `proverbi`): forniti dal corso.
- **Titoli Wikipedia IT**: `itwiki-latest-all-titles-in-ns0`
  (https://dumps.wikimedia.org/itwiki/latest/).
- **Corpus Paisà** per le collocazioni: https://www.corpusitaliano.it/ . Le
  associazioni si generano una volta con:
  ```bash
  python src/mine_corpus.py --corpus /percorso/paisa.raw.utf8 --out data/corpus_assoc.tsv
  ```
- **LLM per la descrizione**: [Ollama](https://ollama.com) con `ollama pull qwen2.5:7b`.

I percorsi si passano via variabili d'ambiente (`WIKI_TITLES_PATH`,
`CORPUS_ASSOC_PATH`, `IT_EMBEDDINGS_PATH`).

## Esecuzione

```bash
cd src

# Solo ranker (soluzione), con titoli + corpus
WIKI_TITLES_PATH=/path/itwiki-latest-all-titles-in-ns0 \
CORPUS_ASSOC_PATH=../data/corpus_assoc.tsv \
  python run_baseline.py

# Pipeline completa: soluzione + descrizione (Ollama attivo)
WIKI_TITLES_PATH=/path/itwiki-latest-all-titles-in-ns0 \
CORPUS_ASSOC_PATH=../data/corpus_assoc.tsv \
  python run_full.py --engine ollama --select classic --out ../results/pred_final.json

# Valutazione: accuracy, tempi, ROUGE/BLEU/BERTScore
python run_eval.py --pred ../results/pred_final.json
```

## Riferimenti

- P. Basile, M. de Gemmis, P. Lops, G. Semeraro. *Solving a Complex Language Game by
  Using Knowledge-Based Word Associations Discovery.* IEEE TCIAIG, 2016.
- F. Sangati, A. Pascucci, J. Monti. *Exploiting Multiword Expressions to solve "La
  Ghigliottina".* / *Il Mago della Ghigliottina* @ EVALITA 2018/2020.
- P. Bojanowski et al. *Enriching Word Vectors with Subword Information* (fastText).
- T. Zhang et al. *BERTScore: Evaluating Text Generation with BERT.* ICLR 2020.

## Licenza

Codice rilasciato sotto licenza MIT (vedi [`LICENSE`](LICENSE)). I dati usati
(dataset del corso, liste lessicali, corpus Paisà, fastText, modelli) mantengono le
rispettive licenze e non sono inclusi nel repository.
