# Risultati

Test set ufficiale: 100 partite. Parametri di scoring selezionati su un dev set di
200 partite del training; il test è usato solo per la valutazione finale.

## Soluzione

Contributo cumulativo di ciascuna fonte della knowledge base (scoring per copertura
+ penalità di rarità, senza embeddings):

| Knowledge base | Acc@1 | MRR | Recall candidati |
|---|---|---|---|
| MWE + proverbi | 25% | 0.30 | 63% |
| + titoli Wikipedia | 36% | 0.43 | 89% |
| **+ collocazioni da corpus (Paisà, PMI≥3)** | **47%** | **0.55** | 89% |

Tempo medio di risoluzione: 0.01 s/partita; nessuna partita oltre il limite di 60 s.

Progressione delle scelte di design: conteggio MWE grezzo 21% → scoring per copertura
+ rarità 25% → titoli Wikipedia 36% → collocazioni da corpus 47%.

## Descrizione

Generata da Qwen2.5-7B (locale, via Ollama), 100 descrizioni, ~7.5 s/descrizione:

| ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore-F1 |
|---|---|---|---|---|
| 0.330 | 0.065 | 0.199 | 6.73 | 0.733 |

I valori di ROUGE/BLEU sono contenuti perché circa metà delle descrizioni è generata
per una soluzione predetta errata (accuracy 47%) e anche quelle corrette
parafrasano il riferimento in modo diverso. Il BERTScore multilingue, più tollerante,
si attesta a 0.73.

## Ablation: componenti provate e scartate

- **Embeddings (fastText cc.it.300).** Utili per il recall prima dell'aggiunta dei
  titoli Wikipedia (Acc@1 25%→31%), ma ridondanti e leggermente dannosi una volta
  inseriti titoli e corpus; non fanno parte del sistema finale.
- **Scorer a grafo (Personalized PageRank).** La propagazione di attivazione dai 5
  indizi non supera il ranking diretto: il collo di bottiglia è la copertura delle
  associazioni, non l'algoritmo di scoring.
- **Selezione della soluzione tramite LLM.** Peggiora il ranker knowledge-based:
  modelli locali (Qwen2.5-7B/14B) vicini allo 0% in un test diagnostico, con il 14B
  spesso oltre il limite di 60 s; un LLM general-purpose (GPT-4) si ferma intorno al
  4% Acc@1 in letteratura. L'LLM è quindi usato solo per la descrizione.

## Note

- Risultati deterministici: i pareggi di punteggio sono risolti in ordine alfabetico,
  identici su qualsiasi macchina.
- Il filtraggio delle stopword e la penalità di rarità (∝ 1/√freq) impediscono alle
  parole molto comuni (es. "fare", "dire") di dominare il ranking.
- Riferimento allo stato dell'arte: il sistema UNIOR4NLP/Sangati dichiara 68.6% Acc@1
  su un dataset EVALITA ridotto, usando pattern sintattici su corpus più grandi.
