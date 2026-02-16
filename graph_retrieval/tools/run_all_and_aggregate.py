import argparse
import os
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
try:
    from zoneinfo import ZoneInfo  # Python ≥3.9
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Python 3.8


from src.tools.yaps_metrics import compute_metrics_sweep_for_eval_csv_wide



# ---------------------------------------------------
# CONFIG: embeddings and which models to run per one
# ---------------------------------------------------
EMBEDDINGS = {
    "d2v": "mdl.emb.d2v.D2v_d2v",
    "n2v": "mdl.emb.gnn.Gnn_n2v",
    "m2v": "mdl.emb.gnn.Gnn_m2v",
    "sage": "mdl.emb.gnn.Gnn_gs",
    "gcn": "mdl.emb.gnn.Gnn_gcn",
}

MODELS = {
    "d2v": ["mdl.fnn.Fnn", "mdl.bnn.Bnn"],
    "n2v": ["mdl.fnn.Fnn", "mdl.bnn.Bnn", "mdl.emb.gnn.Gnn"],
    "m2v": ["mdl.fnn.Fnn", "mdl.bnn.Bnn", "mdl.emb.gnn.Gnn"],
    "sage": ["mdl.fnn.Fnn", "mdl.bnn.Bnn", "mdl.emb.gnn.Gnn"],
    "gcn": ["mdl.fnn.Fnn", "mdl.bnn.Bnn", "mdl.emb.gnn.Gnn"],
}


# ---------------------------------------------------
# ARGPARSE
# ---------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--run_exp", type=str, default="true",
                   help="Run experiments before aggregation (true/false)")
    p.add_argument("--input_dir", type=str, required=True,
                   help="Input data folder (creator_details.csv, yap_scores.csv, etc.)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Base OpeNTF output folder")
    p.add_argument("--verbose", type=str, default="false",
                   help="Show full Hydra + OpeNTF logs (true/false)")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Folder containing cache files like teamsvecs.pkl, splits*.pkl, indexes.pkl")
    p.add_argument("--eval_topk", type=str, default="2,3,5",
                   help="Top-k values for evaluation, e.g. 2,3,5")
    p.add_argument("--min_test_team_size", type=int, default=None,
                   help="Minimum number of members a team must have to be eligible for the test set")

    # >>> NEW: disable sweep (only lambda=0.0)
    p.add_argument("--disable_yaps_sweep", type=str, default="false",
                   help="If true, compute only lambda=0.0 (true/false)")

    return p.parse_args()


def parse_ks(eval_topk: str):
    ks = []
    for x in str(eval_topk).split(","):
        x = x.strip()
        if x:
            ks.append(int(x))
    if not ks:
        raise ValueError(f"--eval_topk is empty/invalid: {eval_topk}")
    return ks


# ---------------------------------------------------
# RUN A SINGLE OpeNTF EXPERIMENT
# ---------------------------------------------------
def run_opentf_experiment(
    embedding_name, embedding_class, model_list, eval_topk,
    min_test_team_size, input_dir, output_dir, cache_dir=None
):
    print("\n" + "=" * 80)
    print(f"🔥 RUNNING EMBEDDING: {embedding_name}  ->  {embedding_class}")
    print(f"🔥 MODELS: {model_list}")
    print("=" * 80 + "\n")

    model_str = ",".join(model_list)

    if embedding_name == "d2v":
        cmd_seq = "cmd=[prep,train,test,eval]"
    else:
        cmd_seq = "cmd=[train,test,eval]"

    ks = parse_ks(eval_topk)
    max_k = max(ks)
    test_topK = max(max_k, 20)

    cmd = [
        "/content/conda/bin/python",
        "src/main.py",
        cmd_seq,
        f"models.instances=[{model_str}]",
        "data.domain=cmn.web3project.Web3Project",
        f'data.source="{input_dir}"',
        f"data.output={output_dir}",
        f"data.embedding.class_method={embedding_class}",
        "acceleration='cpu'",
        "data.acceleration='cpu'",
        "models.config=src/mdl/__config__.yaml",
        "data.embedding.config=src/mdl/emb/__config__.yaml",
        "~data.filter",
        "train.save_per_epoch=15",
        f"eval.topk=\\'{eval_topk}\\'",
        "test.per_epoch=true",
        f"test.topK={test_topK}",
        f"test.min_test_team_size={min_test_team_size}",
    ]

    if cache_dir is not None:
        cmd.append(f'+data.cache_dir="{cache_dir}"')

    cmd_str = " ".join(cmd)
    print(cmd_str)
    subprocess.run(cmd_str, shell=True, check=False)


# ---------------------------------------------------
# COLLECT test.pred.eval.mean.csv FILES UNDER A SPLITS DIR
# (We use them as anchors to locate the run directory.
# Metrics are recomputed from f0.test.pred + cache files.)
# ---------------------------------------------------
def collect_eval_files_for_split(split_dir):
    files = []
    for root, _, fnames in os.walk(split_dir):
        for f in fnames:
            if f == "test.pred.eval.mean.csv":
                parent = os.path.dirname(os.path.join(root, f))
                if os.path.normpath(parent) == os.path.normpath(split_dir):
                    continue
                files.append(os.path.join(root, f))
    return sorted(files)


# ---------------------------------------------------
# PARSE RUN INFO FROM PATH (relative to split dir)
# ---------------------------------------------------
def parse_run_info(path, split_dir):
    rel = os.path.relpath(path, split_dir)
    parts = rel.split(os.sep)

    trial_id = None
    if parts and parts[0].startswith("trial_"):
        trial_id = int(parts[0].split("_", 1)[1])
        parts = parts[1:]

    if len(parts) == 2:
        emb_dir = parts[0]
        model_dir = emb_dir
        embedding_short = emb_dir.split(".")[0]
        model_short = "gnn"
    elif len(parts) >= 3:
        emb_dir = parts[0]
        model_dir = parts[1]
        embedding_short = emb_dir.split(".")[0]
        model_short = model_dir.split(".")[0]
    else:
        return {
            "run_dir": rel,
            "embedding_setting": "",
            "model_setting": "",
            "embedding_short": "",
            "model_short": "",
            "trial_id": trial_id,
        }

    run_dir_parts = [os.path.basename(split_dir)]
    if trial_id is not None:
        run_dir_parts.append(f"trial_{trial_id}")
    run_dir_parts.extend(parts[:-1])

    return {
        "run_dir": os.path.join(*run_dir_parts),
        "embedding_setting": emb_dir,
        "model_setting": model_dir,
        "embedding_short": embedding_short,
        "model_short": model_short,
        "trial_id": trial_id,
    }

def safe_to_csv(df: pd.DataFrame, path: str, **kwargs):
    df = df.copy()

    # force plain Python objects
    df.columns = pd.Index([str(c) for c in df.columns], dtype=object)
    df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

    return df.to_csv(path, **kwargs)


# ---------------------------------------------------
# SAVE: one row per (model × lambda) with metrics as columns
# ---------------------------------------------------
def save_metrics_summary_over_trials_wide(df_trials_wide: pd.DataFrame, analysis_dir: str, ts: str):
    """
    Input df_trials_wide rows:
      (trial_id, embedding_short, model_short, yaps_lambda, metric columns...)

    Output: metrics_summary_over_trials.csv
      One row per (embedding/model/lambda) aggregated over trials,
      columns: <metric>_mean and <metric>_std
    """
    id_cols = [
        "embedding_setting",
        "model_setting",
        "embedding_short",
        "model_short",
        "yaps_lambda",
    ]

    metric_cols = [c for c in df_trials_wide.columns if c not in (id_cols + ["trial_id", "run_dir"])]

    # mean/std across trials for each group
    agg_mean = df_trials_wide.groupby(id_cols, as_index=False)[metric_cols].mean()
    agg_std  = df_trials_wide.groupby(id_cols, as_index=False)[metric_cols].std(ddof=1)

    # rename columns
    mean_ren = {c: f"{c}_mean" for c in metric_cols}
    std_ren  = {c: f"{c}_std" for c in metric_cols}

    agg_mean = agg_mean.rename(columns=mean_ren)
    agg_std  = agg_std.rename(columns=std_ren)

    # merge mean + std
    out = agg_mean.merge(agg_std, on=id_cols, how="left")
    out["n_trials"] = df_trials_wide.groupby(id_cols)["trial_id"].nunique().values
    out["run_timestamp"] = ts

    # order columns
    front = ["run_timestamp", "yaps_lambda", "embedding_setting", "model_setting", "embedding_short", "model_short", "n_trials"]
    other = [c for c in out.columns if c not in front]
    out = out[front + other]

    out_path = os.path.join(analysis_dir, "metrics_summary_over_trials.csv")
    safe_to_csv(out, out_path, index=False)
    print(f"📊 Saved: {out_path}")
    return out


# ---------------------------------------------------
# WINNER TABLE (optional): best per metric across (model×lambda)
# ---------------------------------------------------
def build_winner_table_from_summary(df_summary_over_trials: pd.DataFrame, analysis_dir: str):
    """
    df_summary_over_trials is the aggregated wide file (mean/std columns).
    We pick best row per metric_mean column.
    """
    metric_mean_cols = [c for c in df_summary_over_trials.columns if c.endswith("_mean")]

    winners = []
    for col in metric_mean_cols:
        # skip NaNs
        tmp = df_summary_over_trials.dropna(subset=[col])
        if tmp.empty:
            continue
        idx = tmp[col].idxmax()
        row = tmp.loc[idx]
        winners.append({
            "metric": col.replace("_mean", ""),
            "best_mean": float(row[col]),
            "yaps_lambda": float(row["yaps_lambda"]),
            "embedding_short": row["embedding_short"],
            "model_short": row["model_short"],
            "embedding_setting": row["embedding_setting"],
            "model_setting": row["model_setting"],
            "n_trials": int(row["n_trials"]),
        })

    winners_df = pd.DataFrame(winners)
    out_path = os.path.join(analysis_dir, "winners_table.csv")
    safe_to_csv(winners_df, out_path, index=False)
    print(f"🏆 Saved: {out_path}")
    return winners_df


# ---------------------------------------------------
# BUILD SUMMARY FOR ONE SPLIT
# ---------------------------------------------------
def build_summary_for_split(split_dir, analysis_dir, ts, input_dir, cache_dir, ks, disable_yaps_sweep=False):
    eval_files = collect_eval_files_for_split(split_dir)
    if not eval_files:
        print(f"⚠️ No test.pred.eval.mean.csv files found under {split_dir}.")
        return None

    if cache_dir is None:
        raise ValueError("cache_dir is required (needs teamsvecs.pkl + splits*.pkl + indexes.pkl).")

    # internal lambdas (NO CLI)
    LAMBDAS = [0.0] if disable_yaps_sweep else [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]

    all_trial_rows = []

    print(f"📂 Found {len(eval_files)} runs under {split_dir}.")
    for fpath in eval_files:
        run_info = parse_run_info(fpath, split_dir)

        try:
            # >>> returns a DataFrame with one row per lambda (WIDE)
            df_wide = compute_metrics_sweep_for_eval_csv_wide(
                eval_csv_path=fpath,
                cache_dir=cache_dir,
                input_dir=input_dir,
                ks=ks,
                lambdas=LAMBDAS,
                trial_id=(run_info["trial_id"] if run_info["trial_id"] is not None else 0),
                verbose=False,
)
           



        except Exception as e:
            print(f"⚠️ Sweep failed for:\n   {fpath}\n   reason: {e}")
            continue

        # add run identifiers to each lambda row
        df_wide["run_dir"] = run_info["run_dir"]
        df_wide["embedding_setting"] = run_info["embedding_setting"]
        df_wide["model_setting"] = run_info["model_setting"]
        df_wide["embedding_short"] = run_info["embedding_short"]
        df_wide["model_short"] = run_info["model_short"]
        df_wide["trial_id"] = run_info["trial_id"]
        df_wide.columns = df_wide.columns.map(str)   # avoid pandas Index bug.to_csv
        trial_id = run_info["trial_id"]
        trial_tag = f"trial_{trial_id}" if trial_id is not None else "trial_0"
        
        fname = (
            f"{trial_tag}_"
            f"{run_info['embedding_short']}_"
            f"{run_info['model_short']}_"
            "yaps_sweep.csv"
        )
        
        

        df_wide = df_wide.copy()
        df_wide.columns = pd.Index([str(c) for c in df_wide.columns], dtype=object)
        
        safe_to_csv(df_wide, os.path.join(analysis_dir, fname), index=False)




        all_trial_rows.append(df_wide)

    if not all_trial_rows:
        print("⚠️ No runs could be processed.")
        return None

    df_trials = pd.concat(all_trial_rows, ignore_index=True)

    # keep raw per-trial+lambda file (debug)
    raw_path = os.path.join(analysis_dir, "all_results_raw_trials_wide.csv")
    safe_to_csv(df_trials, raw_path, index=False)
    print(f"🧾 Saved raw trial-wide rows: {raw_path}")

    # aggregate across trials -> one row per (model × lambda)
    df_summary = save_metrics_summary_over_trials_wide(df_trials, analysis_dir, ts)
    build_winner_table_from_summary(df_summary, analysis_dir)
    return df_summary


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    args = parse_args()
    ks = parse_ks(args.eval_topk)

    run_experiments = args.run_exp.lower() == "true"
    disable_yaps_sweep = args.disable_yaps_sweep.lower() == "true"

    ts = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y-%m-%d_%H-%M")

    # output_dir selection
    BASE_OUTPUT = args.output_dir
    if run_experiments:
        exp_name = f"run_{ts}"
        output_dir = os.path.join(BASE_OUTPUT, exp_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Experiment output directory:\n   {output_dir}\n")
    else:
        output_dir = BASE_OUTPUT
        print(f"Aggregating existing results from:\n   {output_dir}\n")

    # verbose hydra
    if args.verbose.lower() == "true":
        print("🔍 VERBOSE MODE ENABLED — full Hydra logs will be shown.\n")
        os.environ["HYDRA_FULL_ERROR"] = "1"
        os.environ["HYDRA_VERBOSITY"] = "debug"
        os.environ["HYDRA_LOG_LEVEL"] = "debug"
    else:
        print("ℹ️ Verbose mode OFF — only standard logs will be shown.\n")

    # run experiments
    if run_experiments:
        for emb_name, emb_class in EMBEDDINGS.items():
            run_opentf_experiment(
                embedding_name=emb_name,
                embedding_class=emb_class,
                model_list=MODELS[emb_name],
                eval_topk=args.eval_topk,
                min_test_team_size=args.min_test_team_size,
                input_dir=args.input_dir,
                output_dir=output_dir,
                cache_dir=args.cache_dir,
            )
    else:
        print("⚙️ Skipping experiments (run_exp=false).")

    # detect splits
    splits_dirs = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("splits.") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not splits_dirs:
        print("ℹ️ No splits.* folders found — aggregating directly under the run folder.")
        splits_dirs = [output_dir]

    print("\n📁 Split folders:")
    for sd in splits_dirs:
        print(f"   - {sd}")

    # aggregate per split
    for split_dir in splits_dirs:
        split_name = os.path.basename(split_dir)
        analysis_dir = os.path.join(split_dir, f"analysis_{ts}")
        os.makedirs(analysis_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"🗂 Analysis for split: {split_name}")
        print(f"📁 Analysis folder: {analysis_dir}")
        print("=" * 80 + "\n")

        build_summary_for_split(
            split_dir=split_dir,
            analysis_dir=analysis_dir,
            ts=ts,
            input_dir=args.input_dir,
            cache_dir=args.cache_dir,
            ks=ks,
            disable_yaps_sweep=disable_yaps_sweep,
        )

    print("\n✅ All splits processed.")


if __name__ == "__main__":
    main()
