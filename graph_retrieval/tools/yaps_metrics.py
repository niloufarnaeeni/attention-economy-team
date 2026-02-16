# src/tools/yaps_metrics.py

import os
import re
import math
import zipfile
import tempfile
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------
# IMPORTANT:
# We MUST use calculate_metrics and calculate_skill_coverage
# from src.eval.metric.py (your requirement).
# ------------------------------------------------------------
try:
    # Adjust this import if your package layout differs.
    # Common possibilities:
    # from src.eval.metric import calculate_metrics, calculate_skill_coverage
    # from eval.metric import calculate_metrics, calculate_skill_coverage
    from src.evl.metric import calculate_metrics, calculate_skill_coverage
except Exception as e:
    raise ImportError(
        "Could not import calculate_metrics/calculate_skill_coverage from src.eval.metric.py. "
        "Fix the import path in src/tools/yaps_metrics.py.\n"
        f"Original error: {e}"
    )

# -----------------------------
# Helpers (same as before)
# -----------------------------
def _norm(s: str) -> str:
    return str(s).strip().lower().lstrip("@")


def _strip_numeric_prefix(code: str) -> str:
    """
    '0_c3' -> 'c3'
    'c3' -> 'c3'
    'C4' -> 'c4'
    """
    code = _norm(code)
    m = re.match(r"^\d+_(.+)$", code)
    return m.group(1) if m else code


def load_indexes_pkl(
    cache_dir: Path,
    fallback_dir: Optional[Path] = None
) -> Dict:
    cache_dir = Path(cache_dir)
    p = cache_dir / "indexes.pkl"

    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)

    if fallback_dir is not None:
        fallback_dir = Path(fallback_dir)
        p2 = fallback_dir / "indexes.pkl"
        if p2.exists():
            with open(p2, "rb") as f:
                return pickle.load(f)

    raise FileNotFoundError(f"Missing indexes.pkl. Expected at: {p}")


def normalize_indexes_i2c(indexes: Dict) -> Dict:
    if "i2c" not in indexes:
        return indexes

    i2c = indexes["i2c"]

    if isinstance(i2c, (list, tuple)):
        indexes["i2c"] = [_strip_numeric_prefix(x) for x in i2c]
    else:
        try:
            indexes["i2c"] = {
                k: _strip_numeric_prefix(v) for k, v in i2c.items()
            }
        except Exception:
            pass

    return indexes


def load_creator_yaps(input_dir: Path) -> Dict[str, float]:
    """
    Reads:
      - input_dir/yap_scores.csv columns: username,yaps_all
      - input_dir/creator_details.csv columns: Creator_ID,creator_code

    Returns:
      map creator_code(lower/normalized) -> yaps_all
    """
    input_dir = Path(input_dir)

    yaps_file = input_dir / "yap_scores.csv"
    creators_file = input_dir / "creator_details.csv"

    if not yaps_file.exists():
        raise FileNotFoundError(f"Missing file: {yaps_file}")
    if not creators_file.exists():
        raise FileNotFoundError(f"Missing file: {creators_file}")

    yaps_df = pd.read_csv(yaps_file)
    creators_df = pd.read_csv(creators_file)

    if not {"username", "yaps_all"}.issubset(set(yaps_df.columns)):
        raise KeyError(
            f"{yaps_file} must contain columns {{'username','yaps_all'}}. "
            f"Found: {list(yaps_df.columns)}"
        )

    if not {"Creator_ID", "creator_code"}.issubset(set(creators_df.columns)):
        raise KeyError(
            f"{creators_file} must contain columns {{'Creator_ID','creator_code'}}. "
            f"Found: {list(creators_df.columns)}"
        )

    yaps_df["username_norm"] = yaps_df["username"].astype(str).map(_norm)
    creators_df["creator_id_norm"] = creators_df["Creator_ID"].astype(str).map(_norm)

    merged = creators_df.merge(
        yaps_df[["username_norm", "yaps_all"]],
        left_on="creator_id_norm",
        right_on="username_norm",
        how="left",
    )

    yaps_map: Dict[str, float] = {}

    for _, row in merged.iterrows():
        val = row["yaps_all"]
        if pd.isna(val):
            continue

        key = _strip_numeric_prefix(str(row["creator_code"]))
        yaps_map[key] = float(val)

    return yaps_map


# ------------------------------------------------------------
# Locate f0.test.pred (predictions)
# ------------------------------------------------------------
def find_pred_zip_for_eval_csv(
    eval_csv_path: str,
    max_search_up: int = 8
) -> Tuple[str, str]:
    """
    Find f0.test.pred near the eval csv path.
    Returns: (pred_zip_path, fold_tag)
    """
    eval_dir = Path(eval_csv_path).resolve().parent
    target_name = "f0.test.pred"
    fold_tag = "f0.test"

    local = eval_dir / target_name
    if local.exists() and local.is_file():
        return str(local), fold_tag

    cur = eval_dir
    for _ in range(max_search_up + 1):
        cand = cur.parent / target_name
        if cand.exists() and cand.is_file():
            return str(cand), fold_tag
        cur = cur.parent

    raise FileNotFoundError(
        f"Required pred file '{target_name}' not found near eval_csv={eval_csv_path}"
    )


def load_y_pred_from_pred_zip(
    pred_zip_path: str,
    fold_tag: str
) -> torch.Tensor:
    """
    Loads y_pred from f0.test.pred (zip file)
    and returns sparse COO tensor [B,E]
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(pred_zip_path, "r") as zf:
            members = [m for m in zf.namelist() if m.startswith(fold_tag)]
            if not members:
                raise ValueError(f"'{fold_tag}' not found in {pred_zip_path}")

            zf.extractall(temp_dir, members)

        extracted_root = os.path.join(temp_dir, fold_tag)
        repacked = os.path.join(temp_dir, "repacked.pth")

        with zipfile.ZipFile(repacked, "w") as zout:
            for root, _, files in os.walk(extracted_root):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, extracted_root)
                    zout.write(full, os.path.join("archive", rel))

        data = torch.load(
            repacked,
            map_location="cpu",
            weights_only=False
        )

        if "y_pred" not in data:
            raise KeyError(
                f"'y_pred' missing in {pred_zip_path}, keys={list(data.keys())}"
            )

        y_pred = data["y_pred"]
        if y_pred.layout != torch.sparse_coo:
            y_pred = y_pred.to_sparse_coo()

        return y_pred.coalesce()


# ------------------------------------------------------------
# Load teamsvecs.pkl + splits*.pkl
# ------------------------------------------------------------
def _find_one(cache_dir: Path, patterns: List[str]) -> Path:
    for pat in patterns:
        hits = list(Path(cache_dir).glob(pat))
        if hits:
            hits = sorted(hits, key=lambda p: str(p))
            return hits[0]

    raise FileNotFoundError(
        f"Could not find any file in {cache_dir} matching patterns={patterns}"
    )


def load_ground_truth_and_skillcoverage(
    cache_dir: Path,
    trial_id: int
):
    """
    Returns:
      Y_test: scipy sparse [B,E]
      X_skill: scipy sparse [B,S]
      expertskillvecs: scipy sparse [E,S]
    """
    cache_dir = Path(cache_dir)

    teamsvecs_path = _find_one(cache_dir, ["teamsvecs.pkl"])
    splits_path = _find_one(cache_dir, ["splits*.pkl", "splits.*.pkl"])

    with open(teamsvecs_path, "rb") as f:
        teamsvecs = pickle.load(f)

    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    if "member" not in teamsvecs:
        raise KeyError("teamsvecs.pkl missing key 'member'")

    if "trials" not in splits:
        raise KeyError("splits*.pkl missing key 'trials'")

    if trial_id not in splits["trials"]:
        raise KeyError(
            f"trial_id={trial_id} not in splits['trials']. "
            f"Available: {list(splits['trials'].keys())}"
        )

    test_idx = splits["trials"][trial_id]["test"]
    Y_test = teamsvecs["member"][test_idx]

    if "skill" in teamsvecs:
        X_skill = teamsvecs["skill"][test_idx]
    elif "original_skill" in teamsvecs:
        X_skill = teamsvecs["original_skill"][test_idx]
    else:
        raise KeyError(
            "teamsvecs.pkl must contain 'skill' or 'original_skill'"
        )

    expertskillvecs = teamsvecs.get("skillcoverage", None)

    return Y_test, X_skill, expertskillvecs


# ------------------------------------------------------------
# Convert sparse y_pred -> dense scores
# ------------------------------------------------------------
def sparse_to_dense_scores(
    y_pred_sparse: torch.Tensor,
    fill_value: float = -1e9
) -> np.ndarray:
    """
    y_pred_sparse: torch sparse COO [B,E]
    returns dense numpy array [B,E]
    """
    y_pred_sparse = y_pred_sparse.coalesce()
    B, E = int(y_pred_sparse.size(0)), int(y_pred_sparse.size(1))

    out = np.full((B, E), fill_value, dtype=np.float32)

    idx = y_pred_sparse.indices().cpu().numpy()
    val = y_pred_sparse.values().cpu().numpy()

    out[idx[0], idx[1]] = val.astype(np.float32)

    return out


# ------------------------------------------------------------
# YAPS exposure metrics
# ------------------------------------------------------------
def yaps_exposure_from_dense(
    Y_scores: np.ndarray,
    indexes: Dict,
    yaps_map: Dict[str, float],
    ks: List[int],
) -> Dict[str, float]:
    """
    AvgYaps@k / DiscYaps@k based on ranking induced by Y_scores
    """
    if "i2c" not in indexes:
        raise KeyError("indexes missing 'i2c'")

    i2c = indexes["i2c"]
    E = Y_scores.shape[1]

    yaps_vec = np.zeros(E, dtype=np.float64)
    for j in range(E):
        code = i2c[j] if isinstance(i2c, (list, tuple)) else i2c[j]
        code = _strip_numeric_prefix(code)
        yaps_vec[j] = float(yaps_map.get(code, 0.0))

    out: Dict[str, float] = {}

    for k in ks:
        avg_vals = []
        disc_vals = []

        topk_idx = np.argpartition(
            -Y_scores,
            kth=min(k, E) - 1,
            axis=1
        )[:, :min(k, E)]

        for i in range(Y_scores.shape[0]):
            cand = topk_idx[i]
            cand = cand[np.argsort(Y_scores[i, cand])[::-1]]
            cand = cand[:k]

            y = yaps_vec[cand]
            if y.size == 0:
                continue

            avg_vals.append(float(y.mean()))
            disc_vals.append(
                float(
                    np.sum([
                        y[t] / math.log2(t + 2)
                        for t in range(len(y))
                    ])
                )
            )

        out[f"AvgYaps@{k}"] = float(np.mean(avg_vals)) if avg_vals else 0.0
        out[f"DiscYaps@{k}"] = float(np.mean(disc_vals)) if disc_vals else 0.0

    return out


# ------------------------------------------------------------
# Metric name expansion
# ------------------------------------------------------------
def expand_metric_names(ks: List[int]) -> List[str]:
    m = []
    for k in ks:
        m.append(f"P_{k}")
    for k in ks:
        m.append(f"recall_{k}")
    for k in ks:
        m.append(f"ndcg_cut_{k}")
    for k in ks:
        m.append(f"map_cut_{k}")
    return m


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def compute_metrics_sweep_for_eval_csv_wide(
    eval_csv_path: str,
    *,
    cache_dir: str,
    input_dir: str,
    ks: List[int],
    lambdas: List[float],
    trial_id: int,
    max_search_up: int = 8,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Returns a WIDE DataFrame with one row per lambda
    """
    pred_zip_path, fold_tag = find_pred_zip_for_eval_csv(
        eval_csv_path,
        max_search_up=max_search_up
    )

    y_pred = load_y_pred_from_pred_zip(pred_zip_path, fold_tag)

    indexes = normalize_indexes_i2c(
        load_indexes_pkl(Path(cache_dir))
    )

    yaps_map = load_creator_yaps(Path(input_dir))

    Y_test, X_skill, expertskillvecs = load_ground_truth_and_skillcoverage(
        Path(cache_dir),
        trial_id=trial_id
    )

    base_scores = sparse_to_dense_scores(y_pred, fill_value=-1e9)

    if "i2c" not in indexes:
        raise KeyError("indexes missing 'i2c'")

    i2c = indexes["i2c"]
    E = base_scores.shape[1]

    yaps_bias = np.zeros(E, dtype=np.float32)
    for j in range(E):
        code = i2c[j] if isinstance(i2c, (list, tuple)) else i2c[j]
        code = _strip_numeric_prefix(code)
        y = float(yaps_map.get(code, 0.0))
        yaps_bias[j] = float(math.log1p(y))

    metric_names = expand_metric_names(ks)
    topK = max(ks)

    rows = []

    for lam in lambdas:
        lam = float(lam)

        Y_scores = base_scores + lam * yaps_bias[None, :]

        _, df_mean = calculate_metrics(
            Y_test,
            Y_scores,
            topK=topK,
            per_instance=False,
            metrics=metric_names,
        )

        metric_dict = {
            str(idx): float(v)
            for idx, v in df_mean["mean"].items()
        }

        if expertskillvecs is not None:
            _, df_skc_mean = calculate_skill_coverage(
                X_skill,
                Y_scores,
                expertskillvecs,
                per_instance=False,
                topks=",".join(str(k) for k in ks),
            )
            skc_dict = {
                str(idx): float(v)
                for idx, v in df_skc_mean["mean"].items()
            }
        else:
            skc_dict = {
                f"skill_coverage_{k}": np.nan
                for k in ks
            }

        metric_dict.update(skc_dict)

        yexp = yaps_exposure_from_dense(
            Y_scores=Y_scores,
            indexes=indexes,
            yaps_map=yaps_map,
            ks=ks,
        )

        metric_dict.update(yexp)

        row = {"yaps_lambda": lam}
        row.update(metric_dict)
        rows.append(row)

    df = pd.DataFrame(rows)

    front = ["yaps_lambda"]
    rest = [c for c in df.columns if c not in front]
    df = df[front + sorted(rest)]

    return df
