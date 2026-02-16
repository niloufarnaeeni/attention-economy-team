import json
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
from rag_retrieval import Reranker
import math
from pathlib import Path

# -----------------------------
# Regex helpers
# -----------------------------
_CREATOR_RE = re.compile(r"\bCreator\s+(C\d+)\b", re.IGNORECASE)

def norm(x):
    return x.lower().lstrip("@")


def load_creator_yaps(raw_dir):
    yaps_df = pd.read_csv(raw_dir / "yap_scores.csv")
    creators_df = pd.read_csv(raw_dir / "creator_details.csv")

    # normalize
    yaps_df["username_norm"] = yaps_df["username"].str.lower().str.lstrip("@")
    creators_df["creator_id_norm"] = creators_df["Creator_ID"].str.lower().str.lstrip("@")

    merged = creators_df.merge(
        yaps_df,
        left_on="creator_id_norm",
        right_on="username_norm",
        how="left",
    )

    # final mapping: C4 → yaps_all
    yaps_map = {
        row["creator_code"]: float(row["yaps_all"])
        for _, row in merged.iterrows()
        if not pd.isna(row["yaps_all"])
    }

    return yaps_map


def extract_creator_id(doc_text: str) -> Optional[str]:
    """
    Expects doc text to contain: 'Creator C169 ...'
    Returns 'C169' or None.
    """
    if not doc_text:
        return None
    m = _CREATOR_RE.search(doc_text)
    return m.group(1) if m else None


def extract_project_name_from_query(query: str) -> Optional[str]:
    """
    Works with your query like:
      'Creators suitable for the Injective project.'
    Returns: 'Injective'
    """
    if not query:
        return None
    m = re.search(
        r"Creators\s+suitable\s+for\s+the\s+(.+?)\s+project",
        query,
        re.IGNORECASE,
    )
    return m.group(1).strip().strip(".") if m else None


def build_qrel_and_run_from_grouped_jsonl(
    jsonl_path: str,
    ranker: Reranker,
    k: Optional[int] = None,
    relevance_if_label_gt: float = 0.0,
    creator_yaps: Optional[Dict[str, float]] = None,
    yaps_lambda: float = 0.0,
):


    """
    Build qrels + run from grouped reranker data:

      {"query": str, "hits": [{"content": str, "label": float}, ...], ...}

    qrels (binary, for P@k/Recall@k):
      relevant iff label > relevance_if_label_gt (default: 0.0)

    run:
      docid -> predicted score from ranker

    Returns:
      qrels: {qid: {docid: 1}}
      run  : {qid: {docid: score}}
    """
    qrels: Dict[str, Dict[str, int]] = {}
    run: Dict[str, Dict[str, float]] = {}
    qid_to_query = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)

            query = rec["query"]
            hits = rec["hits"]

            docs: List[str] = [h["content"] for h in hits]
            labels: List[float] = [float(h["label"]) for h in hits]

            # Make qid unique and readable
            project = extract_project_name_from_query(query) or f"q{idx}"
            trial_id = rec.get("trial_id", 0)
            fold_id = rec.get("fold_id", 0)
            split = rec.get("split", "unknown")
            qid = f"{project}|t{trial_id}|f{fold_id}|{split}|i{idx:04d}"
            qid_to_query[qid] = query
            
            # -----------------------------
            # Reranker inference
            # -----------------------------
            ranked = ranker.rerank(query, docs)
            results = ranked.results[:k] if k else ranked.results

            # -----------------------------
            # RUN: predicted scores
            # -----------------------------
            run[qid] = {}
            for r in results:
                docid = extract_creator_id(r.text) or f"DOC_{r.doc_id}"
                base_score = float(r.score)

                if creator_yaps is not None:
                    yaps = creator_yaps.get(docid, 0.0)
                    base_score = base_score + yaps_lambda * math.log1p(yaps)
                    #base_score = base_score - yaps_lambda * math.log1p(yaps)
                
                run[qid][docid] = base_score



            # -----------------------------
            # QRELS: binary relevance for P@k
            # -----------------------------
            qrels[qid] = {}
            for doc_text, lab in zip(docs, labels):
                if lab > relevance_if_label_gt:
                    docid = extract_creator_id(doc_text)
                    if docid:
                        qrels[qid][docid] = 1
                    # If creator id is missing, skip (safer than mismatched ids)

    return qrels, run, qid_to_query 
