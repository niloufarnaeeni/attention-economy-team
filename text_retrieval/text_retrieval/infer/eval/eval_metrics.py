import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
import pytrec_eval
import math


# -----------------------------
# Measures
# -----------------------------
def make_measures_for_ks(ks=(2, 5, 10)) -> set:
    """
    Measures that match your current reporting needs.
    make_measures_for_ks defines TREC-native metrics that:

    are computed by pytrec_eval

    must be known to pytrec_eval.RelevanceEvaluator
    """
    measures = set()
    for k in ks:
        measures |= {f"P_{k}", f"recall_{k}", f"ndcg_cut_{k}", f"map_cut_{k}",}
    return measures


def evaluate_once(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    ks=(2, 5, 10),
    extra_measures: Optional[List[str]] = None,
) -> Tuple[set, Dict[str, Dict[str, float]]]:
    measures = make_measures_for_ks(ks)
    if extra_measures:
        measures |= set(extra_measures)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_query = evaluator.evaluate(run)
    return measures, per_query


def aggregate_metrics(per_query: Dict[str, Dict[str, float]], measures: set) -> dict:
    agg = {}
    for m in measures:
        vals = [per_query[qid].get(m) for qid in per_query if m in per_query[qid]]
        agg[m] = float(sum(vals) / len(vals)) if vals else 0.0
    agg["num_q"] = float(len(per_query))
    return agg

#def compute_yaps_metrics(run, creator_yaps, ks):
#    avg = {}
#    disc = {}
#    if creator_yaps is None:
#        return {}, {}

#    for k in ks:
#        avg_vals = []
#        disc_vals = []

#        for qid, docs in run.items():
#            ranked = sorted(docs.items(), key=lambda x: x[1], reverse=True)[:k]

#            yaps_vals = [creator_yaps.get(docid, 0.0) for docid, _ in ranked]
#            if not yaps_vals:
#                continue

#            avg_vals.append(sum(yaps_vals) / len(yaps_vals))

#            disc_score = sum(
#                y / math.log2(i + 2) for i, y in enumerate(yaps_vals)
#            )
 #           disc_vals.append(disc_score)
#
#       avg[f"AvgYaps@{k}"] = sum(avg_vals) / len(avg_vals) if avg_vals else 0.0
 #       disc[f"DiscYaps@{k}"] = sum(disc_vals) / len(disc_vals) if disc_vals else 0.0

  #  return avg, disc
import math

def compute_yaps_metrics(run, creator_yaps, ks):
    """
    Paper-consistent exposure metrics using g(x)=log(1+x):

      Exp@k  := log(1 + (1/k) * sum_{i=1..k} a_i)
      DExp@k := log(1 + sum_{i=1..k} a_i / log2(i+1))

    We store them into existing columns:
      AvgYaps@k  (as Exp@k)
      DiscYaps@k (as DExp@k)
    """
    avg = {}
    disc = {}
    if creator_yaps is None:
        return {}, {}

    for k in ks:
        exp_vals = []
        dexp_vals = []

        for qid, docs in run.items():
            ranked = sorted(docs.items(), key=lambda x: x[1], reverse=True)[:k]
            a_vals = [float(creator_yaps.get(docid, 0.0)) for docid, _ in ranked]

            # If fewer than k docs exist, pad with zeros to keep 1/k semantics
            if len(a_vals) < k:
                a_vals += [0.0] * (k - len(a_vals))

            # Exp@k = g( (1/k) * sum a_i )
            mean_a = sum(a_vals) / k
            exp_vals.append(math.log1p(mean_a))

            # DExp@k = g( sum a_i / log2(i+1) )
            disc_sum = sum(
                a / math.log2(i + 2) for i, a in enumerate(a_vals)  # i=0 -> log2(2)=1
            )
            dexp_vals.append(math.log1p(disc_sum))

        avg[f"AvgYaps@{k}"] = sum(exp_vals) / len(exp_vals) if exp_vals else 0.0
        disc[f"DiscYaps@{k}"] = sum(dexp_vals) / len(dexp_vals) if dexp_vals else 0.0

    return avg, disc


def add_yaps_metrics_to_agg(agg, run, creator_yaps, ks, yaps_lambda=0.0, use_yaps=False):
    """
    Compute Yaps exposure metrics and merge them into agg.
    IMPORTANT:
      - use_yaps means: Yaps affected the ranking score
      - Yaps metrics are ALWAYS computed if creator_yaps exists
    """
    # use_yaps must reflect SCORING, not existence
    agg["use_yaps"] = bool(use_yaps)
    agg["yaps_lambda"] = yaps_lambda if use_yaps else 0.0

    # Yaps metrics are always computed (as you want)
    if creator_yaps is None:
        return agg

    avg_yaps, disc_yaps = compute_yaps_metrics(run, creator_yaps, ks)
    agg.update(avg_yaps)
    agg.update(disc_yaps)
    return agg



# -----------------------------
# Saving CSVs
# -----------------------------
def save_metrics_csv(
    per_query: Dict[str, Dict[str, float]],
    model_name: str,
    meta: Optional[dict],
    output_csv: str,
    ks=(2, 5, 10),
) -> None:
    """
    Save per-query metrics table.
    """
    rows = []
    for qid, metrics in per_query.items():
        row = {
            "query_id": qid,
            "model": model_name,
        }
        if meta:
            row.update({
                "template_version": meta.get("template_name", "grouped"),
                "has_neg": meta.get("has_neg", False),
                "has_score": meta.get("has_score", True),
            })
        else:
            row.update({
                "template_version": "grouped",
                "has_neg": False,
                "has_score": True,
            })

        # Add metrics in your preferred format: P@k, R@k, NDCG@k, MAP@
        for k in ks:
            row[f"P@{k}"] = metrics.get(f"P_{k}", 0.0)
            row[f"R@{k}"] = metrics.get(f"recall_{k}", 0.0)
            row[f"NDCG@{k}"] = metrics.get(f"ndcg_cut_{k}", 0.0)
            row[f"MAP@{k}"] = metrics.get(f"map_cut_{k}", 0.0)
            row[f"SkillCoverage@{k}"] = metrics.get(f"skill_coverage_{k}", 0.0)
            


        # If you include extra measures like map/recip_rank, store them too
        #if "map" in metrics:
        #    row["map"] = metrics.get("map", 0.0)
        if "recip_rank" in metrics:
            row["mrr"] = metrics.get("recip_rank", 0.0)

        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Per-query metrics saved to {output_csv}")


def append_agg_metrics_csv(
    agg: dict,
    ks: tuple,
    output_csv: str,
) -> None:
    """
    Append one aggregated row per run to output_csv.
    
    """
   

    row = {
    "run_time": agg.get("run_time", "unknown"),
    "model_name": agg.get("model_name_or_path", "unknown"),
    "loss_type": agg.get("loss_type", "unknown"),
    "skill_set_version": agg.get("skill_set_version", "unknown"),
    "template_version": agg.get("template_name", "unknown"),
    "ks": ",".join(map(str, ks)),
    "num_q": agg.get("num_q", 0.0),
    "use_yaps": agg.get("use_yaps", False),
    "yaps_lambda": agg.get("yaps_lambda", 0.0),
    "training_lambda": agg.get("training_lambda", "na"),

    }



    # metrics in strict order
    for k in ks:
        row[f"P_{k}"] = agg.get(f"P_{k}", 0.0)
    for k in ks:
        row[f"recall_{k}"] = agg.get(f"recall_{k}", 0.0)
    for k in ks:
        row[f"ndcg_cut_{k}"] = agg.get(f"ndcg_cut_{k}", 0.0)
    for k in ks:
        row[f"map_cut_{k}"] = agg.get(f"map_cut_{k}", 0.0)  
    for k in ks:
        row[f"skill_coverage_{k}"] = agg.get(f"skill_coverage_{k}", 0.0)
    for k in ks:
        row[f"AvgYaps@{k}"] = agg.get(f"AvgYaps@{k}", 0.0)
        row[f"DiscYaps@{k}"] = agg.get(f"DiscYaps@{k}", 0.0)

    
    if "recip_rank" in agg:
        row["recip_rank"] = agg.get("recip_rank", 0.0)

    df_new = pd.DataFrame([row])

    if os.path.exists(output_csv):
        df_old = pd.read_csv(output_csv)
    
        # preserve old column order
        old_cols = list(df_old.columns)
        new_cols = [c for c in df_new.columns if c not in old_cols]
    
        # append new columns at the END
        df = pd.concat([df_old, df_new], ignore_index=True)
    
        # explicitly enforce column order
        df = df[old_cols + new_cols]
    else:
        df = df_new


    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"📈 Aggregated metrics appended to {output_csv}")

# -----------------------------
# Debug / sanity helpers
# -----------------------------
def analyze_run_binary(qrels: dict, run: dict) -> None:
    """
    Works with binary qrels (label>0 -> 1).
    Helpful sanity checks.
    """
    num_q = len(run)
    if num_q == 0:
        print("⚠️ No queries in run.")
        return

    docs_per_q = []
    rel_in_run = []
    nonrel_in_run = []
    zero_hit = 0

    for qid in run:
        retrieved = set(run[qid].keys())
        relevant = set(qrels.get(qid, {}).keys())

        docs_per_q.append(len(retrieved))
        rel_count = len(retrieved & relevant)
        nonrel_count = len(retrieved - relevant)

        rel_in_run.append(rel_count)
        nonrel_in_run.append(nonrel_count)

        if rel_count == 0:
            zero_hit += 1

    print("⚙️ RUN ANALYSIS (binary qrels)")
    print(f"  Queries evaluated         : {num_q}")
    print(f"  Avg docs / query          : {sum(docs_per_q)/num_q:.2f}")
    print(f"  Avg relevant retrieved    : {sum(rel_in_run)/num_q:.2f}")
    print(f"  Avg non-relevant retrieved: {sum(nonrel_in_run)/num_q:.2f}")
    print(f"  Queries with 0 relevant   : {zero_hit} ({100*zero_hit/num_q:.1f}%)")
    print("")
    

def load_run_config_or_default(model_arg: str) -> dict:
    """
    Load run_config.json ONLY from <model_arg>/run_config.json.

    If model_arg is not a local path or the file does not exist,
    return defaults.
    """
    defaults = {
        "template_name": "v2",
        "loss_type": "unknown",
        "model_name_or_path": model_arg,
    }

    model_path = Path(model_arg)

    # HF model id → not a path
    if not model_path.exists():
        return defaults

    cfg_path = model_path / "run_config.json"
    if not cfg_path.exists():
        return defaults

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return {**defaults, **cfg}  # cfg overrides defaults
    except Exception:
        return defaults

