"""Valutazione completa a partire dal file predictions.json prodotto da run_full.py.

Metriche soluzione: accuracy, tempo medio, partite oltre 60s.
Metriche descrizione: ROUGE-1/2/L, BLEU, BERTScore (modello BERT multilingue).

Uso:
    python run_eval.py
    python run_eval.py --pred results/predictions.json

bert-score scarica al primo uso il modello multilingue (bert-base-multilingual-cased);
se non disponibile (offline), la parte BERTScore viene saltata con un avviso.
"""
from __future__ import annotations

import argparse
import json

import config


def eval_solutions(preds: list[dict]) -> dict:
    n = len(preds)
    correct = sum(p["correct"] for p in preds)
    # tempo di RISOLUZIONE (il limite 60s vale su questo, non sulla descrizione)
    solve_times = [p.get("solve_time_s", p.get("elapsed_s", 0.0)) for p in preds]
    desc_times = [p.get("desc_time_s", 0.0) for p in preds]
    over = sum(p.get("over_60s", False) for p in preds)
    # le partite oltre 60s contano come NON risolte (da traccia)
    correct_within_limit = sum(
        p["correct"] and not p.get("over_60s", False) for p in preds
    )
    return {
        "accuracy_raw": correct / n,
        "accuracy_official": correct_within_limit / n,  # con vincolo 60s
        "avg_solve_time_s": sum(solve_times) / n,
        "max_solve_time_s": max(solve_times) if solve_times else 0.0,
        "avg_desc_time_s": sum(desc_times) / n,
        "games_over_60s": over,
    }


def eval_descriptions(preds: list[dict]) -> dict:
    pairs = [
        (p["pred_desc"], p["gold_desc"])
        for p in preds
        if p.get("pred_desc") and p.get("gold_desc")
    ]
    if not pairs:
        return {"note": "nessuna descrizione da valutare"}
    hyps, refs = [p for p, _ in pairs], [r for _, r in pairs]
    out: dict = {"n_descriptions": len(pairs)}

    # ROUGE
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for h, r in zip(hyps, refs):
            s = scorer.score(r, h)
            for k in agg:
                agg[k] += s[k].fmeasure
        out.update({k: v / len(pairs) for k, v in agg.items()})
    except ImportError:
        out["rouge_note"] = "rouge-score non installato"

    # BLEU (corpus)
    try:
        import sacrebleu
        out["bleu"] = sacrebleu.corpus_bleu(hyps, [refs]).score
    except ImportError:
        out["bleu_note"] = "sacrebleu non installato"

    # BERTScore (modello multilingue, richiesto dalla traccia)
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(
            hyps, refs, model_type="bert-base-multilingual-cased", lang="it",
            rescale_with_baseline=False, verbose=False,
        )
        out["bertscore_f1"] = float(F1.mean())
        out["bertscore_p"] = float(P.mean())
        out["bertscore_r"] = float(R.mean())
    except Exception as e:  # noqa: BLE001
        out["bertscore_note"] = f"BERTScore saltato: {e}"

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(config.RESULTS_DIR / "predictions.json"))
    args = ap.parse_args()

    with open(args.pred, encoding="utf-8") as f:
        preds = json.load(f)

    sol = eval_solutions(preds)
    desc = eval_descriptions(preds)

    print("=== SOLUZIONE ===")
    print(f"Accuracy (raw)            : {100*sol['accuracy_raw']:.1f}%")
    print(f"Accuracy ufficiale (<60s) : {100*sol['accuracy_official']:.1f}%")
    print(f"Tempo medio risoluzione   : {sol['avg_solve_time_s']:.2f}s (max {sol['max_solve_time_s']:.2f}s)")
    print(f"Tempo medio descrizione   : {sol['avg_desc_time_s']:.2f}s")
    print(f"Partite oltre 60s         : {sol['games_over_60s']}")
    print("\n=== DESCRIZIONE ===")
    for k, v in desc.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    out_path = config.RESULTS_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"solution": sol, "description": desc}, f, ensure_ascii=False, indent=2)
    print(f"\nMetriche salvate in {out_path}")


if __name__ == "__main__":
    main()
