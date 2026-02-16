import os
import json
import ast
import pickle
import random
import logging
from pathlib import Path
import pandas as pd
import argparse
from typing import List, Any, Dict, Optional, Tuple

from rag_retrieval.helpers.template_utils import load_templates, load_template_metadata
from rag_retrieval.train.reranker.builders.preprocess import preprocess


log = logging.getLogger(__name__)


# -------------------------
# Small utilities
# -------------------------
def _norm_key(x: Any) -> str:
    return str(x).strip().lower()


def _safe_list_parse(raw: Any) -> List[str]:
    """
    Parses list-like fields such as:
      "['S1','S2']" or "['C1', 'C2']"
    and falls back to simple comma-split if needed.
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, set)):
            return [str(x).strip() for x in v if str(x).strip()]
        return [str(v).strip()] if str(v).strip() else []
    except Exception:
        s2 = s.strip().strip("[]")
        parts = [p.strip().strip("'").strip('"') for p in s2.split(",")]
        return [p for p in parts if p]


def t_from_rank(rank: int, n_pos: int) -> float:
    """
    positives: t = (Npos - r + 1) / Npos   => best=1, worst=1/Npos
    negatives: 0 (handled outside)
    """
    if n_pos <= 0:
        raise ValueError("n_pos must be >= 1")
    if not (1 <= rank <= n_pos):
        raise ValueError(f"rank out of range: rank={rank}, n_pos={n_pos}")
    return (n_pos - rank + 1) / n_pos


def _load_splits_bundle(splits_pkl: str) -> Dict:
    """
    Supports BOTH formats:
      - bundle: {'trials': {tid: {...}}, 'n_trials': K}
      - legacy: single split dict: {'test':..., 'folds':...}
    Returns a bundle always: {'trials': {0: split}, 'n_trials': 1} if legacy.
    """
    with open(splits_pkl, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "trials" in obj:
        return obj

    # legacy -> wrap
    return {"trials": {0: obj}, "n_trials": 1}


def _load_indexes(indexes_pkl: str) -> Dict:
    with open(indexes_pkl, "rb") as f:
        return pickle.load(f)


def _maybe_load_teamsvecs_shape(teamsvecs_pkl: str) -> Optional[Tuple[int, int]]:
    """
    Returns (n_teams, n_members) if possible, else None.
    Only used for sanity checks.
    """
    try:
        with open(teamsvecs_pkl, "rb") as f:
            tv = pickle.load(f)
        n_teams = tv["skill"].shape[0]
        n_members = tv["member"].shape[1]
        return n_teams, n_members
    except Exception:
        return None


def _row_indices_to_project_names(
    row_indices: List[int],
    indexes: Dict,
    skills_df: pd.DataFrame,
) -> List[str]:
    """
    row_idx (split) -> project_id (indexes['i2t']) -> project_name (skills.csv)
    """
    if "i2t" not in indexes:
        raise KeyError("indexes.pkl is missing 'i2t' (row_idx -> team_id mapping).")

    # project_id -> project_name (canonical)
    if "project_id" not in skills_df.columns or "project_name" not in skills_df.columns:
        raise KeyError("skills.csv must contain columns: project_id, project_name")

    pid_to_name = {}
    for _, r in skills_df.iterrows():
        try:
            pid = int(r["project_id"])
            pname = str(r["project_name"]).strip()
            if pname:
                pid_to_name[pid] = pname
        except Exception:
            continue

    out = []
    missing = 0
    for idx in row_indices:
        idx = int(idx)

        # indexes['i2t'] might be dict {row_idx: team_id}
        if isinstance(indexes["i2t"], dict):
            if idx not in indexes["i2t"]:
                missing += 1
                continue
            project_id = int(indexes["i2t"][idx])
        else:
            # or list-like
            project_id = int(indexes["i2t"][idx])

        pname = pid_to_name.get(project_id)
        if not pname:
            missing += 1
            continue
        out.append(pname)

    if missing:
        log.warning(f"[Mapping] {missing} split indices could not be mapped to a project_name (skipped).")

    # dedupe while preserving order
    seen = set()
    deduped = []
    for x in out:
        k = _norm_key(x)
        if k not in seen:
            seen.add(k)
            deduped.append(x)
    return deduped



def load_creator_yaps(raw_dir: Path):
    yaps_file = raw_dir / "yap_scores.csv"
    creators_file = raw_dir / "creator_details.csv"

    if not yaps_file.exists():
        raise FileNotFoundError(f"Missing Yaps file: {yaps_file}")

    if not creators_file.exists():
        raise FileNotFoundError(f"Missing creator details file: {creators_file}")

    yaps_df = pd.read_csv(yaps_file)
    creators_df = pd.read_csv(creators_file)

    # normalize
    yaps_df["username_norm"] = yaps_df["username"].str.lower().str.lstrip("@")
    creators_df["creator_id_norm"] = creators_df["Creator_ID"].str.lower().str.lstrip("@")

    merged = creators_df.merge(
        yaps_df,
        left_on="creator_id_norm",
        right_on="username_norm",
        how="left",
    )

    # final mapping: creator_code → yaps_all
    yaps_map = {
        row["creator_code"]: float(row["yaps_all"])
        for _, row in merged.iterrows()
        if not pd.isna(row["yaps_all"])
    }

    return yaps_map



# -------------------------
# Core JSONL builder
# -------------------------
def build_grouped_jsonl_for_projects(
    *,
    projects: List[str],
    skills_csv: str,
    creators_csv: str,
    gpt5_skills_csv: str,
    creator_details_csv: str,
    output_jsonl: str,
    template_version: str,
    neg_ratio: int = 3,
    seed: int = 42,
    use_yaps: bool = False,
    record_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Builds GroupedRankerDataset-style JSONL:
      {"query": "...", "hits": [{"content": "...", "label": float}, ...], ...meta }
    """
    random.seed(seed)
    record_meta = record_meta or {}

    # Templates
    QUERY_TEMPLATE, DOC_TEMPLATE = load_templates(template_version)
    meta = load_template_metadata(template_version)
    HAS_NEG = meta.get("has_neg", True)

    # CSVs
    skills_df = pd.read_csv(skills_csv)
    creators_df = pd.read_csv(creators_csv)
    skill_desc_df = pd.read_csv(gpt5_skills_csv)
    creator_rank_df = pd.read_csv(creator_details_csv)
    
    yaps_map = {}
    if use_yaps:
        yaps_map = load_creator_yaps(Path(os.path.dirname(creator_details_csv)))

    # skill_id -> description
    if "id" not in skill_desc_df.columns or "one_sentence_definition" not in skill_desc_df.columns:
        raise KeyError("gpt5_skills_csv must contain columns: id, one_sentence_definition")

    skill_id_to_desc = {
        str(row["id"]).strip(): str(row["one_sentence_definition"]).strip()
        for _, row in skill_desc_df.iterrows()
    }

    # Build robust mappings keyed by normalized project name
    # skills.csv: project_name -> skill_ids
    if "project_name" not in skills_df.columns or "assigned_skill_ids" not in skills_df.columns:
        raise KeyError("skills_csv must contain columns: project_name, assigned_skill_ids")

    skills_map = {}
    canonical_name = {}
    for _, row in skills_df.iterrows():
        pname = str(row["project_name"]).strip()
        if not pname:
            continue
        key = _norm_key(pname)
        canonical_name[key] = pname
        skills_map[key] = _safe_list_parse(row["assigned_skill_ids"])

    # creators.csv: project_name -> creators
    if "project_name" not in creators_df.columns or "creators" not in creators_df.columns:
        raise KeyError("creators_csv must contain columns: project_name, creators")

    creators_map = {}
    for _, row in creators_df.iterrows():
        pname = str(row["project_name"]).strip()
        if not pname:
            continue
        key = _norm_key(pname)
        creators_map[key] = _safe_list_parse(row["creators"])

    # ranks table
    need_cols = {"creator_code", "project_name", "Rank"}
    if not need_cols.issubset(set(creator_rank_df.columns)):
        raise KeyError(f"creator_details_csv must contain columns: {sorted(need_cols)}")

    rank_df = creator_rank_df[["creator_code", "project_name", "Rank"]].copy()

    # normalize project names in creator_details.csv to match pkey
    rank_df["project_name_norm"] = rank_df["project_name"].astype(str).map(_norm_key)


    # negatives pool
    all_creators = sorted(rank_df["creator_code"].dropna().astype(str).unique())

    written = 0
    skipped = 0

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for project in projects:
            pkey = _norm_key(project)

            # Use canonical name from skills.csv if possible
            proj_name = canonical_name.get(pkey, project)

            if pkey not in skills_map:
                skipped += 1
                continue
            if pkey not in creators_map:
                skipped += 1
                continue

            project_skills = skills_map[pkey]

            # ---- query ----
            skill_desc_text = "\n".join(
                f"{sid}: {skill_id_to_desc.get(str(sid).strip(), 'N/A')}"
                for sid in project_skills
            )
            query = QUERY_TEMPLATE.format(
                project_name=proj_name,
                skill_ids=", ".join(project_skills),
                skill_descriptions=skill_desc_text,
            ).strip()

            # ---- positives ----
            participant_creators = set(creators_map[pkey])

            df_pos = rank_df[
                    (rank_df["project_name_norm"] == pkey)
                    & (rank_df["creator_code"].astype(str).isin(participant_creators))
                ].copy()


            if df_pos.empty:
                log.warning(f"[Skip] No positives found for project='{proj_name}' (pkey='{pkey}'). "
                            f"Check creator_details.csv project_name mapping.")
                skipped += 1
                continue


            # sort by Rank and re-rank to 1..N_pos
            df_pos["Rank"] = pd.to_numeric(df_pos["Rank"], errors="coerce")
            df_pos = df_pos.dropna(subset=["Rank"])
            if df_pos.empty:
                skipped += 1
                continue

            df_pos = df_pos.sort_values("Rank", ascending=True).reset_index(drop=True)
            df_pos["new_rank"] = range(1, len(df_pos) + 1)

            n_pos = len(df_pos)
            hits = []

            for _, row in df_pos.iterrows():
                r = int(row["new_rank"])
                label = float(t_from_rank(r, n_pos))

                doc = DOC_TEMPLATE.format(
                    creator_id=str(row["creator_code"]),
                    project_name=proj_name,
                    skill_ids=", ".join(project_skills),
                ).strip()

                hit = {
                    "content": doc,
                    "label": label,
                }
                
                if use_yaps:
                    hit["yap_score"] = float(yaps_map.get(row["creator_code"], 0.0))
                
                hits.append(hit)



            # ---- negatives ----
            if HAS_NEG and neg_ratio > 0:
                pos_creators = set(df_pos["creator_code"].astype(str).tolist())
                candidates = [c for c in all_creators if c not in pos_creators]
                random.shuffle(candidates)

                n_neg = min(len(candidates), neg_ratio * n_pos)
                for neg_creator in candidates[:n_neg]:
                    neg_doc = DOC_TEMPLATE.format(
                        creator_id=str(neg_creator),
                        project_name=proj_name,
                        skill_ids=", ".join(project_skills),
                    ).strip()
                    hit = {
                        "content": neg_doc,
                        "label": 0.0,
                    }
                    
                    if use_yaps:
                        hit["yap_score"] = float(yaps_map.get(neg_creator, 0.0))
                    
                    hits.append(hit)



            record = {"query": query, "hits": hits, **record_meta}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[Grouped JSONL] wrote={written}, skipped={skipped}, path={output_jsonl}")


def build_one_trial_one_fold(
    *,
    splits_pkl: str,
    indexes_pkl: str,
    teamsvecs_pkl: Optional[str],
    trial_id: int,
    fold_id: int,
    out_train_jsonl: str,
    out_valid_jsonl: str,
    skills_csv: str,
    creators_csv: str,
    gpt5_skills_csv: str,
    creator_details_csv: str,
    template_version: str,
    use_yaps: bool = False,
    neg_ratio: int = 3,
    seed: int = 42,
) -> None:
    # Load bundle splits
    bundle = _load_splits_bundle(splits_pkl)
    trials = bundle["trials"]
    trial = trials[int(trial_id)]
    fold = trial["folds"][int(fold_id)]

    # Load indexes
    indexes = _load_indexes(indexes_pkl)

    # Optional sanity check with teamsvecs
    if teamsvecs_pkl:
        shape = _maybe_load_teamsvecs_shape(teamsvecs_pkl)
        if shape is not None:
            n_teams, n_members = shape
            if isinstance(indexes.get("i2t"), dict):
                i2t_len = len(indexes["i2t"])
            else:
                i2t_len = len(indexes["i2t"])
            if i2t_len != n_teams:
                log.warning(f"[Sanity] indexes.i2t length ({i2t_len}) != teamsvecs n_teams ({n_teams}).")

    # Read skills.csv once for mapping project_id -> project_name
    skills_df = pd.read_csv(skills_csv)

    train_idx = list(map(int, fold["train"]))
    valid_idx = list(map(int, fold["valid"]))
    test_idx = []

    # fold-level test (if exists)
    if "test" in fold:
        test_idx = list(map(int, fold["test"]))
    
    # trial-level test fallback (legacy format)
    elif "test" in trial:
        test_idx = list(map(int, trial["test"]))


    train_projects = _row_indices_to_project_names(train_idx, indexes, skills_df)
    valid_projects = _row_indices_to_project_names(valid_idx, indexes, skills_df)
    test_projects = _row_indices_to_project_names(test_idx, indexes, skills_df)


    print(f"[Split] trial={trial_id} fold={fold_id} train={len(train_projects)} valid={len(valid_projects)}")

    build_grouped_jsonl_for_projects(
        projects=train_projects,
        skills_csv=skills_csv,
        creators_csv=creators_csv,
        gpt5_skills_csv=gpt5_skills_csv,
        creator_details_csv=creator_details_csv,
        output_jsonl=out_train_jsonl,
        template_version=template_version,
        neg_ratio=neg_ratio,
        use_yaps=use_yaps,
        seed=seed,
        record_meta={"trial_id": int(trial_id), "fold_id": int(fold_id), "split": "train"},
    )

    build_grouped_jsonl_for_projects(
        projects=valid_projects,
        skills_csv=skills_csv,
        creators_csv=creators_csv,
        gpt5_skills_csv=gpt5_skills_csv,
        creator_details_csv=creator_details_csv,
        output_jsonl=out_valid_jsonl,
        template_version=template_version,
        neg_ratio=neg_ratio,
        use_yaps=use_yaps,
        seed=seed,
        record_meta={"trial_id": int(trial_id), "fold_id": int(fold_id), "split": "valid"},
    )
    
    if test_projects:
        out_test_jsonl = os.path.join(os.path.dirname(out_train_jsonl), "test.jsonl")
    
        build_grouped_jsonl_for_projects(
            projects=test_projects,
            skills_csv=skills_csv,
            creators_csv=creators_csv,
            gpt5_skills_csv=gpt5_skills_csv,
            creator_details_csv=creator_details_csv,
            output_jsonl=out_test_jsonl,
            template_version=template_version,
            neg_ratio=neg_ratio,
            use_yaps=use_yaps,
            seed=seed,
            record_meta={
                "trial_id": int(trial_id),
                "fold_id": int(fold_id),
                "split": "test",
            },
        )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build grouped JSONL (train + valid) for one trial & one fold"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to raw data directory (contains skills.csv, creators.csv, splits.pkl, etc.)",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for processed JSONL files",
    )

    parser.add_argument(
        "--trial_id",
        type=int,
        default=0,
        help="Trial index (default: 0)",
    )

    parser.add_argument(
        "--fold_id",
        type=int,
        default=0,
        help="Fold index inside the trial (default: 0)",
    )

    parser.add_argument(
        "--template_version",
        type=str,
        required=True,
        help="Template version name",
    )

    parser.add_argument(
        "--neg_ratio",
        type=int,
        default=3,
        help="Negatives per positive (default: 3)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
    "--use_yaps",
    action="store_true",
    help="If set, load Yaps scores and include them in the dataset",
    )


    return parser.parse_args()

def main():
    args = parse_args()

    data_root = args.data_root
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    splits_pkl   = os.path.join(data_root, "splits.t5.r0.85.pkl")
    indexes_pkl  = os.path.join(data_root, "indexes.pkl")
    teamsvecs_pk = os.path.join(data_root, "teamsvecs.pkl")

    skills_csv   = os.path.join(data_root, "skills.csv")
    creators_csv = os.path.join(data_root, "creators.csv")
    gpt5_csv     = os.path.join(data_root, "gpt5_skills.csv")
    details_csv  = os.path.join(data_root, "creator_details.csv")

    out_train = os.path.join(out_dir, "train.jsonl")
    out_valid = os.path.join(out_dir, "valid.jsonl")
    
    preprocess(data_root)

    build_one_trial_one_fold(
        splits_pkl=splits_pkl,
        indexes_pkl=indexes_pkl,
        teamsvecs_pkl=teamsvecs_pk,
        trial_id=args.trial_id,
        fold_id=args.fold_id,
        out_train_jsonl=out_train,
        out_valid_jsonl=out_valid,
        skills_csv=skills_csv,
        creators_csv=creators_csv,
        gpt5_skills_csv=gpt5_csv,
        creator_details_csv=details_csv,
        template_version=args.template_version,
        use_yaps=args.use_yaps,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

