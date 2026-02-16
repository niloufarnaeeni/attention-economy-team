import argparse
import json
from pathlib import Path
from typing import Optional, List

from rag_retrieval import Reranker

# Same-folder (same package) imports
from .eval_metrics import (
    evaluate_once,
    aggregate_metrics,
    save_metrics_csv,
    append_agg_metrics_csv,
    analyze_run_binary,
    load_run_config_or_default,
    compute_yaps_metrics, 
    add_yaps_metrics_to_agg
)

import pickle

from .skill_coverage import (
    parse_qid_meta,
    get_skipteams_from_splits,
    gen_member_skill_cooccurrence,
    compute_skill_coverage_at_k,
)

from .build_run_qrels import (
    build_qrel_and_run_from_grouped_jsonl,
    load_creator_yaps,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate grouped reranker JSONL with pytrec_eval (P@k, Recall@k, NDCG@k)"
    )
    p.add_argument("--jsonl", required=True, help="Path to grouped dataset JSONL")
    p.add_argument(
        "--model",
        required=True,
        help="HF model id OR local path to model folder (e.g. output/kaito/model)",
    )
    p.add_argument("--ks", default="2,5,10", help="Comma-separated cutoffs, e.g. 2,5,10")
    p.add_argument(
        "--output_dir",
        default="eval_outputs",
        help="Base output dir. Outputs will be written to <output_dir>/eval/",
    )

    # grouped-specific
    p.add_argument(
        "--relevance_gt",
        type=float,
        default=0.0,
        help="Binary relevance threshold: label > relevance_gt is relevant (default: 0.0)",
    )
    p.add_argument(
        "--k_run",
        type=int,
        default=None,
        help="Optional: only keep top-k docs in the run (None keeps all).",
    )

    # optional extras (OFF by default)
    p.add_argument(
        "--extra_measures",
        default="",
        help='Optional extra measures, comma-separated (e.g. "map,recip_rank"). Default: none.',
    )
    p.add_argument(
    "--use_yaps",
    action="store_true",
    help="Enable inference-time Yaps re-scoring",
    )
    p.add_argument(
    "--yaps_lambda",
    type=float,
    default=0.1,
    help="Lambda for inference-time Yaps re-scoring",
    )
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()





def main():
    args = parse_args()
    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())

    eval_dir = Path(args.output_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------
    # Infer raw data paths from jsonl location
    # --------------------------------------------------
    jsonl_path = Path(args.jsonl).resolve()
    
   
    root = jsonl_path.parent.parent
    
    raw_dir = root / "raw"
    
    teamsvecs_pkl = raw_dir / "teamsvecs.pkl"
    indexes_pkl   = raw_dir / "indexes.pkl"
    splits_pkl    = raw_dir / "splits.t5.r0.85.pkl"
    #creator_yaps = None
    #if args.use_yaps:
    creator_yaps = load_creator_yaps(raw_dir)


    # 1) load run config (or defaults)
    run_cfg = load_run_config_or_default(args.model)
    template_name = run_cfg.get("template_name", "v2")
    loss_type = run_cfg.get("loss_type", "unknown")
    training_lambda = run_cfg.get("lambda", None)

    # 2) load reranker
    ranker = Reranker(args.model, verbose=0)

    # 3) build qrels + run
    qrels, run, qid_to_query = build_qrel_and_run_from_grouped_jsonl(
        jsonl_path=args.jsonl,
        ranker=ranker,
        k=args.k_run,
        relevance_if_label_gt=args.relevance_gt,
        creator_yaps=creator_yaps,
        yaps_lambda=args.yaps_lambda if args.use_yaps else 0.0,
    )

    analyze_run_binary(qrels, run)

    extra: Optional[List[str]] = (
        [m.strip() for m in args.extra_measures.split(",") if m.strip()] or None
    )

    measures, per_query = evaluate_once(
    qrels=qrels,
    run=run,
    ks=ks,
    extra_measures=extra,
    )
    
    # --------------------------------------------------
    # Skill coverage (automatic, no CLI)
    # --------------------------------------------------
    with open(teamsvecs_pkl, "rb") as f:
        teamsvecs = pickle.load(f)
    with open(indexes_pkl, "rb") as f:
        indexes = pickle.load(f)
    with open(splits_pkl, "rb") as f:
        splits_bundle = pickle.load(f)
    
    # infer trial / fold / split from qid
    if not run:
        raise RuntimeError("Run is empty — cannot compute skill coverage.")

    any_qid = next(iter(run))

    trial_id, fold_id, split_name = parse_qid_meta(any_qid)
    
    skipteams = get_skipteams_from_splits(
        splits_bundle, trial_id, fold_id, split_name
    )
    
    cache_path = eval_dir / f"skillcoverage_member_skill_co_t{trial_id}_f{fold_id}_{split_name}.pkl"
    
    member_skill_co = gen_member_skill_cooccurrence(
        teamsvecs=teamsvecs,
        cache_path=cache_path,
        skipteams=skipteams,
    )
    
    skc_per_query = compute_skill_coverage_at_k(
        run=run,
        qid_to_query=qid_to_query,
        indexes=indexes,
        member_skill_co=member_skill_co,
        ks=ks,
    )
    
    # merge into per_query
    for qid in per_query:
        per_query[qid].update(skc_per_query.get(qid, {}))
    
    # tell aggregator to include them
    measures |= {f"skill_coverage_{k}" for k in ks}
    
    # NOW aggregate
    agg = aggregate_metrics(per_query, measures)
    
    #agg = add_yaps_metrics_to_agg(
    #    agg=agg,
    #    run=run,
    #    creator_yaps=creator_yaps if args.use_yaps else None,
    #    ks=ks,
    #    yaps_lambda=args.yaps_lambda,
    #)
    agg = add_yaps_metrics_to_agg(
        agg=agg,
        run=run,
        creator_yaps=creator_yaps,
        ks=ks,
        yaps_lambda=args.yaps_lambda,
        use_yaps=args.use_yaps,   
    )



    meta ={}
    # Use chained assignment (=) instead of unpacking (,)
    agg["run_time"] = meta["run_time"] = run_cfg.get("run_time", "unknown")
    agg["model_name_or_path"] = meta["model_name_or_path"] = run_cfg.get("model_name_or_path", args.model)
    agg["template_name"] = meta["template_name"] = template_name
    agg["loss_type"] = meta["loss_type"] = loss_type
    agg["training_lambda"] = meta["training_lambda"] = training_lambda
    agg["skill_set_version"] = meta["skill_set_version"] = run_cfg.get("skill_set_version", "rootdata")
    
    def safe(s: str) -> str:
        return (
            str(s)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )


    # filenames include loss
    run_id = Path(args.model).parent.name

    safe_loss = safe(loss_type)

    per_query_csv = eval_dir / (
             f"metrics_per_query_{run_id}_{meta['model_name_or_path']}_{loss_type}_{meta['skill_set_version']}_{template_name}.csv"
            )

    agg_csv = eval_dir / "results.csv"

    # per-query CSV uses agg as run_info
    save_metrics_csv(
        per_query=per_query,
        model_name=args.model,
        output_csv=str(per_query_csv),
        ks=ks,
        meta=meta,
    )

    # aggregated CSV appends using agg only
    append_agg_metrics_csv(
        agg=agg,
        ks=ks,
        output_csv=str(agg_csv),
    )

    if args.verbose:
        print("\n📊 Aggregated metrics (this run):")
        print(json.dumps(agg, indent=2))
        print(f"\n✅ Saved per-query CSV: {per_query_csv}")
        print(f"✅ Appended agg CSV   : {agg_csv}")


if __name__ == "__main__":
    main()
