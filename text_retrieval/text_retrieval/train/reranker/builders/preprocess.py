from pathlib import Path
import pandas as pd
import logging

log = logging.getLogger(__name__)

def preprocess(data_root: str) -> None:
    """
    Dataset-agnostic preprocessing entry point.

    Dispatches to dataset-specific preprocessors if needed.
    Safe to call for any dataset.
    """
    path_str = str(data_root).lower()

    if "giverep" in path_str:
        preprocess_giverep(data_root)
    elif any(k in path_str for k in ["cookie", "cookie_fun"]):
        preprocess_cookie_fun(data_root)

    else:
        # other datasets: do nothing
        return
    
def preprocess_giverep(data_root: str) -> None:
    """
    Dataset-specific preprocessing for GIVEREP.

    This function:
      1) Checks whether data_root corresponds to 'giverep'
      2) Ensures creator_details.csv has Creator_ID
      3) Generates yap_scores.csv if missing

    Safe to call multiple times.
    Does nothing for non-giverep datasets.
    """

    data_root = Path(data_root)

    # -------------------------------------------------
    # 1. Check dataset name
    # -------------------------------------------------

    log.info("[GIVEREP] Running Giverep preprocessing")

    creator_details = data_root / "creator_details.csv"
    yaps_file = data_root / "yap_scores.csv"

    if not creator_details.exists():
        raise FileNotFoundError(
            f"[GIVEREP] Missing creator_details.csv at {creator_details}"
        )

    df = pd.read_csv(creator_details)

    # -------------------------------------------------
    # 2. Ensure Creator_ID exists
    # -------------------------------------------------
    if "Creator_ID" not in df.columns:
        if "twitter_handle" not in df.columns:
            raise ValueError(
                "[GIVEREP] twitter_handle column required to create Creator_ID"
            )

        df["Creator_ID"] = (
            df["twitter_handle"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.lstrip("@")
        )

        log.info("[GIVEREP] Added Creator_ID column")

    # -------------------------------------------------
    # 3. Generate yap_scores.csv if missing
    # -------------------------------------------------
    if not yaps_file.exists():
        required_cols = {"Creator_ID", "total_engagement"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"[GIVEREP] Missing required columns for YAP generation: {missing}"
            )

        yaps_df = (
            df.groupby("Creator_ID", as_index=False)["total_engagement"]
            .mean()
            .rename(
                columns={
                    "Creator_ID": "username",
                    "total_engagement": "yaps_all",
                }
            )
        )

        yaps_df["username"] = (
            yaps_df["username"]
            .astype(str)
            .str.lower()
            .str.lstrip("@")
        )

        yaps_df["yaps_all"] = (
            pd.to_numeric(yaps_df["yaps_all"], errors="coerce")
            .fillna(0.0)
        )

        yaps_df = yaps_df.sort_values("yaps_all", ascending=False)

        yaps_df.to_csv(yaps_file, index=False)

        log.info(
            f"[GIVEREP] Generated yap_scores.csv with {len(yaps_df)} creators"
        )

    else:
        log.info("[GIVEREP] yap_scores.csv already exists, skipping generation")

    # -------------------------------------------------
    # 4. Save updated creator_details.csv (if modified)
    # -------------------------------------------------
    df.to_csv(creator_details, index=False)

    log.info("[GIVEREP] Preprocessing complete")

def preprocess_cookie_fun(data_root: Path) -> None:
    """
    Dataset-specific preprocessing for COOKIE.FUN.

    This function:
      1) Recomputes Rank within each project using Mindshare
      2) Ensures Creator_ID exists
      3) Generates yap_scores.csv using mean Mindshare per creator

    Safe to call multiple times.
    """


    log.info("[COOKIE] Running cookie.fun preprocessing")
    data_root = Path(data_root)

    creator_details = data_root / "creator_details.csv"
    yaps_file = data_root / "yap_scores.csv"

    if not creator_details.exists():
        raise FileNotFoundError(f"[COOKIE] Missing {creator_details}")

    df = pd.read_csv(creator_details)

    # -------------------------------------------------
    # 1. Normalize Mindshare (strip % if needed)
    # -------------------------------------------------
    if df["Mindshare"].dtype == object:
        df["Mindshare"] = (
            df["Mindshare"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
        )

    # -------------------------------------------------
    # 2. Fix Rank within each project
    # -------------------------------------------------
    df["Rank"] = (
        df.sort_values(
            by=["project_slug", "Mindshare"],
            ascending=[True, False]
        )
        .groupby("project_slug")
        .cumcount()
        + 1
    )

    log.info("[COOKIE] Recomputed Rank within each project using Mindshare")

    # -------------------------------------------------
    # 3. Ensure Creator_ID
    # -------------------------------------------------
    if "Creator_ID" not in df.columns:
        df["Creator_ID"] = (
            df["Handle"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.lstrip("@")
        )

        log.info("[COOKIE] Added Creator_ID column")

    # -------------------------------------------------
    # 4. Generate yap_scores.csv (global creator signal)
    # -------------------------------------------------
    if not yaps_file.exists():
        yaps_df = (
            df.groupby("Creator_ID", as_index=False)["Mindshare"]
            .mean()
            .rename(
                columns={
                    "Creator_ID": "username",
                    "Mindshare": "yaps_all",
                }
            )
        )

        yaps_df["username"] = (
            yaps_df["username"]
            .astype(str)
            .str.lower()
            .str.lstrip("@")
        )

        yaps_df["yaps_all"] = (
            pd.to_numeric(yaps_df["yaps_all"], errors="coerce")
            .fillna(0.0)
        )

        yaps_df = yaps_df.sort_values("yaps_all", ascending=False)

        yaps_df.to_csv(yaps_file, index=False)

        log.info(
            f"[COOKIE] Generated yap_scores.csv with {len(yaps_df)} creators"
        )
    else:
        log.info("[COOKIE] yap_scores.csv already exists, skipping")

    # -------------------------------------------------
    # 5. Save updated creator_details.csv
    # -------------------------------------------------
    df.to_csv(creator_details, index=False)

    log.info("[COOKIE] Preprocessing complete")
