# Esplorazione dei dati — Ghigliottina

## Dataset

| | Train | Test |
|---|---|---|
| Partite | 965 | 100 |
| Con descrizione | 174 (18%) | 100 (100%) |
| Dal gioco da tavolo (`ttg`) | 150 | 0 |
| Indizi per partita | sempre 5 | sempre 5 |
| Soluzioni multi-parola | 0 | 0 |
| Lunghezza descrizione (parole) | min 56 / media 86 / max 115 | min 53 / media 87 / max 124 |

## Osservazioni chiave (impatto sul design)

1. **La soluzione è SEMPRE una singola parola.** → il vocabolario dei candidati può essere una lista di parole singole italiane; niente gestione di espressioni multi-parola come output.
2. **Solo il 18% del train ha la descrizione, ma il 100% del test sì.** → la descrizione va *generata* a inference time. Le 174 descrizioni del train servono come esempi few-shot per l'LLM e come riferimento di stile/lunghezza (~85 parole).
3. **Solo 22/100 soluzioni di test compaiono nel train.** → la memorizzazione dà al massimo il 22% di accuracy: serve vero ragionamento associativo. Buon segnale che il task non è banale.
4. **Le liste di polirematiche/proverbi sono molto informative:** 79/100 soluzioni di test compaiono in almeno una polirematica/proverbio, e in media 3.8 indizi su 5 per partita vi compaiono. → il "bonus polirematica" nello scoring classico è un segnale forte e giustifica l'approccio ibrido.

## Risorse lessicali disponibili

| File | Voci | Note |
|---|---|---|
| `polirematiche` | 4.695 | con categoria grammaticale (Avv, Sos, ...) |
| `demauro.poli` | ~30.975 | dizionario De Mauro delle polirematiche (le prime 8 righe sono numeri da scartare) |
| `proverbi` | 369 | proverbi italiani |

`demauro.poli` è la risorsa più ricca per il bonus di associazione.

## Conseguenze per la pipeline

- **Vocabolario candidati:** parole singole da (a) soluzioni del train, (b) parole che compaiono nelle polirematiche/proverbi, (c) eventualmente un vocabolario italiano generico dagli embeddings.
- **Scoring classico [B]:** similarità embedding indizio↔candidato + bonus se `candidato + indizio` (o `indizio + candidato`) forma una polirematica/proverbio noto.
- **Few-shot per descrizione:** campionare alcune delle 174 descrizioni del train come esempi nel prompt.
- **Metrica di riferimento descrizione:** target ~85 parole, struttura "Soluzione: -legame con indizio1; -legame con indizio2; ...".
