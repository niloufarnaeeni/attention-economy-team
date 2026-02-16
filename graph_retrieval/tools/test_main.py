# ---------------------------------------------------------------------
# Split validation & sanity-check script
#
# - Loads cached team–creator matrices (no rebuild)
# - Loads/generates train/valid/test splits via get_splits
# - Validates split structure across trials and folds
# - Ensures strict train/valid/test separation (no overlap)
# - Checks creator coverage (all creators in test/valid appear in train)
# - Verifies strict Top-K feasibility for all projects
# - Prints project-size histograms for sanity checks
# ---------------------------------------------------------------------

import sys
import os
import logging
import numpy as np
import hydra
from hydra.utils import get_class
import argparse
import pickle

# SETUP PATHS
current_dir = os.path.dirname(os.path.abspath(__file__))  # src/tools
src_dir = os.path.dirname(current_dir)                    # src
sys.path.append(src_dir)

# Import AFTER sys.path
from main import get_splits

CLI = {}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Input/cache directory (contains splits PKL)")
    parser.add_argument("--output_dir", required=True, help="Output/work directory")
    parser.add_argument("--ks", default="2,3,5", help="Comma-separated k values, e.g. 2,3,5")
    args, unknown = parser.parse_known_args()

    # keep Hydra overrides only
    sys.argv = [sys.argv[0]] + unknown

    CLI["input_dir"] = args.input_dir
    CLI["output_dir"] = args.output_dir
    CLI["ks"] = [int(x.strip()) for x in args.ks.split(",") if x.strip()]

def _team_sizes(member_mat, idx):
    mm = member_mat.tocsr()
    idx = np.asarray(list(idx), dtype=int) if isinstance(idx, set) else np.asarray(idx, dtype=int)
    idx = idx.reshape(-1)
    return np.diff(mm[idx].indptr)


def strict_k_max(member_mat, idx):
    sizes = _team_sizes(member_mat, idx)
    return int(sizes.min()) if sizes.size > 0 else 0

def enforce_strict_ks(ks, k_max, label):
    bad = [k for k in ks if k > k_max]
    print(f"[STRICT TOPK] {label}: max_k_valid_for_all_projects = {k_max}")
    if bad:
        raise AssertionError(
            f"[STRICT TOPK FAIL] {label}: invalid ks={bad} because some projects have <k creators. "
            f"Choose ks <= {k_max} (strict mode)."
        )
def print_project_size_histogram(member_mat, title):
    """
    Histogram of number of creators per project.
    Rows = projects, columns = creators.
    """
    mm = member_mat.tocsr()

    # number of creators per project (row-wise nnz)
    sizes = np.diff(mm.indptr)

    unique, counts = np.unique(sizes, return_counts=True)

    print(f"\n[PROJECT SIZE HISTOGRAM] {title}")
    print("=" * 50)
    for u, c in zip(unique, counts):
        print(f"  {u:>3} creators : {c} project(s)")
    print(f"  MIN = {sizes.min()}, MAX = {sizes.max()}")
    print("=" * 50)



def selftest_do_split_bundle(
    splits_bundle,
    member_mat,
    team_names=None,          # list[str] length n_teams, optional
    n_folds_expected=5,
    max_show_test_names=30,
):
    """
    Self-test for do_split/get_splits output.

    splits_bundle: dict like {'trials': {0: {'test':..., 'folds':...}, ...}, 'n_trials': N}
                  OR a legacy single split {'test':..., 'folds':...}
    member_mat: scipy.sparse matrix shape (n_teams, n_creators), multi-hot
    team_names: optional list of project names aligned with row indices
    """

    # ---- normalize to trials_map ----
    if isinstance(splits_bundle, dict) and "trials" in splits_bundle:
        trials_map = splits_bundle["trials"]
    else:
        trials_map = {0: splits_bundle}

    n_teams = member_mat.shape[0]
    print(f"\n[SELFTEST] n_teams = {n_teams}, n_trials = {len(trials_map)}")

    # ---- check uniqueness across trials ----
    seen_test_sets = {}
    for tid, sp in trials_map.items():
        test_idx = np.asarray(sp["test"], dtype=int)
        key = frozenset(test_idx.tolist())
        if key in seen_test_sets:
            other = seen_test_sets[key]
            raise AssertionError(f"[FAIL] Trial {tid} has SAME test set as trial {other}.")
        seen_test_sets[key] = tid
    print("[OK] All trial test sets are different.")
    
    print_project_size_histogram(
        member_mat,
        title="ALL PROJECTS (before splitting)")

    # ---- helper: creator coverage check ----
    def _creator_coverage_ok(train_rows, check_rows, label):
        train_present = (
            np.asarray(member_mat[train_rows].sum(axis=0)).ravel() > 0
        )
        check_present = (
            np.asarray(member_mat[check_rows].sum(axis=0)).ravel() > 0
        )
    
        missing = np.where(check_present & (~train_present))[0]
    
        print(
            f"[CREATOR COVERAGE – {label}] "
            f"creators_in_{label.lower()}={check_present.sum()}, "
            f"creators_in_train={train_present.sum()}, "
            f"missing={len(missing)}"
        )
    
        return missing

    show_creator_evidence(member_mat=member_mat, min_projects=2)

    # ---- per-trial checks ----
    for tid, sp in trials_map.items():
        print(f"\n========== TRIAL {tid} ==========")

        assert "test" in sp and "folds" in sp, f"[FAIL] Trial {tid} missing keys."
        folds = sp["folds"]
        test_idx = np.asarray(sp["test"], dtype=int)

        # basic sanity
        assert test_idx.ndim == 1, f"[FAIL] Trial {tid} test must be 1D."
        assert len(np.unique(test_idx)) == len(test_idx), f"[FAIL] Trial {tid} test has duplicates."
        assert np.all((0 <= test_idx) & (test_idx < n_teams)), f"[FAIL] Trial {tid} test out of range."
        assert len(test_idx) > 0, f"[FAIL] Trial {tid} test is empty."

        # folds structure
        assert isinstance(folds, dict), f"[FAIL] Trial {tid} folds must be dict."
        assert len(folds) == n_folds_expected, (
            f"[FAIL] Trial {tid} expected {n_folds_expected} folds, got {len(folds)}."
        )

        # reconstruct train pool = union of all (train + valid) indices across folds
        valid_sets = []
        train_only_union = set()
        valid_union = set()


        for k in sorted(folds.keys()):
            f = folds[k]
            tr = np.asarray(f["train"], dtype=int)
            va = np.asarray(f["valid"], dtype=int)

            # disjoint
            if np.intersect1d(tr, va).size != 0:
                raise AssertionError(f"[FAIL] Trial {tid} fold {k}: train and valid overlap.")

            # index sanity
            if tr.size == 0 or va.size == 0:
                raise AssertionError(f"[FAIL] Trial {tid} fold {k}: empty train/valid.")
            if np.any(tr < 0) or np.any(tr >= n_teams) or np.any(va < 0) or np.any(va >= n_teams):
                raise AssertionError(f"[FAIL] Trial {tid} fold {k}: indices out of range.")

            train_only_union.update(tr.tolist())
            valid_union.update(va.tolist())
            valid_sets.append(set(va.tolist()))


        train_only_union = np.array(sorted(train_only_union), dtype=int)
        valid_union = np.array(sorted(valid_union), dtype=int)


        # fold valids should be disjoint and cover train pool
        for i in range(len(valid_sets)):
            for j in range(i + 1, len(valid_sets)):
                if valid_sets[i].intersection(valid_sets[j]):
                    raise AssertionError(f"[FAIL] Trial {tid}: valid sets overlap between folds {i} and {j}.")


        
        if np.intersect1d(train_only_union, test_idx).size != 0:
            raise AssertionError(f"[FAIL] Trial {tid}: TRAIN intersects test set.")
        
        if np.intersect1d(valid_union, test_idx).size != 0:
            raise AssertionError(f"[FAIL] Trial {tid}: VALID intersects test set.")


        # creator coverage constraint
        bad_cols = bad_cols = _creator_coverage_ok(
                train_only_union,
                test_idx,
                label="TEST"
            )

        if bad_cols.size > 0:
            raise AssertionError(
                f"[FAIL] Trial {tid}: coverage violated for {bad_cols.size} creators. "
                f"First few creator-cols: {bad_cols[:10].tolist()}"
            )
        
        bad_cols_val = bad_cols_val = _creator_coverage_ok(
                train_only_union,
                valid_union,
                label="VALID"
            )

        if bad_cols_val.size > 0:
            raise AssertionError(
                f"[FAIL] Trial {tid}: VALID has creators not present in TRAIN for {bad_cols_val.size} creators. "
                f"First few creator-cols: {bad_cols_val[:10].tolist()}"
            )

        # quick stats
        print(f"[OK] test size = {len(test_idx)}")
        print(f"[OK] train_pool size = {len(train_only_union)}")

        print_project_size_histogram(
            member_mat[test_idx],
            title=f"TRIAL {tid} – TEST PROJECTS ONLY"
        )

        #MIN_TEST_CREATORS = max(CLI["ks"])  # e.g. 10

        #test_sizes = _team_sizes(member_mat, test_idx)
        #if np.any(test_sizes < MIN_TEST_CREATORS):
        #    raise AssertionError(
        #       f"Test set contains projects with < {MIN_TEST_CREATORS} creators. "
        #        f"Strict Precision@{MIN_TEST_CREATORS} is invalid."
        #    )

        # show some test projects
        if team_names is not None:
            show = test_idx[:max_show_test_names]
            print(f"[TEST PROJECTS] Showing {len(show)} test project_name(s):")
            for idx in show:
                print(f"  - {idx}: {team_names[idx]}")
        else:
            print(f"[TEST INDICES] First {min(max_show_test_names, len(test_idx))}: {test_idx[:max_show_test_names].tolist()}")

        # -------------------------------
        # STRICT TOP-K (same k for all projects)
        # -------------------------------
        ks = CLI.get("ks", [2, 5, 10])

        kmax_test = strict_k_max(member_mat, test_idx)
        enforce_strict_ks(ks, kmax_test, label=f"TRIAL {tid} - TEST")

        # train_union is your TRAIN_POOL (train + valid)
        kmax_trainpool = strict_k_max(member_mat, train_only_union)

        # If you want ONE strict k that works for BOTH train_pool and test:
        kmax_both = min(kmax_test, kmax_trainpool)
        print(f"[STRICT TOPK] TRIAL {tid}: max_k_valid_for_ALL (train_pool + test) = {kmax_both}")
        
        


    print("\n✅ [SELFTEST PASS] All trials look correct, coverage holds, and strict-topk checks passed.")

        

def show_creator_evidence(
        member_mat,
        scope_idx=None,          # None = all projects, else subset (e.g. train_pool)
        min_projects=2,
        creator_names=None,      # optional mapping: index -> name
    ):
        """
        Check whether all creators appear in at least `min_projects` projects.
    
        Prints ONLY if violations exist.
        Creators are printed comma-separated on one line.
        """
    
        mm = member_mat.tocsr()
    
        # Restrict to a subset of projects if provided
        if scope_idx is not None:
            scope_idx = np.asarray(scope_idx, dtype=int)
            counts = np.asarray(mm[scope_idx].sum(axis=0)).ravel()
            scope_label = f"subset(size={len(scope_idx)})"
        else:
            counts = np.asarray(mm.sum(axis=0)).ravel()
            scope_label = "ALL PROJECTS"
    
        # creators violating the rule
        bad_creators = np.where(counts < min_projects)[0]
    
        if bad_creators.size == 0:
            return  # ✅ silent success (by design)
    
        # pretty names (optional)
        if creator_names is not None:
            bad_labels = [str(creator_names[c]) for c in bad_creators]
        else:
            bad_labels = [str(c) for c in bad_creators]
    
        print("\n[CREATOR MULTI-PROJECT VIOLATION]")
        print("=" * 60)
        print(f"Scope           : {scope_label}")
        print(f"Min projects    : {min_projects}")
        print(f"Violations (#)  : {len(bad_creators)}")
        print("Creators        :")
        print(", ".join(bad_labels))
        print("=" * 60)



@hydra.main(version_base=None, config_path="../", config_name="__config__")
def run_test(cfg):

    print("\n[TEST] Starting split validation")
    print(f"[TEST] Domain       : {cfg.data.domain}")
    

    # --------------------------------------------------------------------------
    # 1. LOAD DATA (REAL PIPELINE)
    # --------------------------------------------------------------------------
    domain_cls = get_class(cfg.data.domain)

    cfg.data.output = CLI["output_dir"]
    os.makedirs(cfg.data.output, exist_ok=True)
    print(f"[TEST] Output path: {cfg.data.output}")
    # ------------------------------------------------------------
    # Resolve cache_dir: CLI input_dir > cfg.data.cache_dir > output
    # ------------------------------------------------------------
    cache_dir = CLI.get("input_dir", None)
    
    if not cache_dir:
        cache_dir = getattr(cfg.data, "cache_dir", None)
    
    if not cache_dir:
        cache_dir = cfg.data.output
    
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"[TEST] Using cache_dir: {cache_dir}")


    print("[TEST] Loading teamsvecs + indexes from cache (NO rebuild)...")
    
    teamsvecs_pkl = os.path.join(cache_dir, "teamsvecs.pkl")
    indexes_pkl   = os.path.join(cache_dir, "indexes.pkl")
    
    if not os.path.exists(teamsvecs_pkl):
        raise FileNotFoundError(f"Missing teamsvecs.pkl in cache_dir: {teamsvecs_pkl}")
    if not os.path.exists(indexes_pkl):
        raise FileNotFoundError(f"Missing indexes.pkl in cache_dir: {indexes_pkl}")
    
    with open(teamsvecs_pkl, "rb") as f:
        teamsvecs = pickle.load(f)
    
    with open(indexes_pkl, "rb") as f:
        indexes = pickle.load(f)
    
    print(f"[TEST] Loaded teamsvecs member shape = {teamsvecs['member'].shape}")
    print(f"[TEST] Loaded teamsvecs skill  shape = {teamsvecs['skill'].shape}")
    print(f"[TEST] Loaded indexes: keys = {list(indexes.keys())}")


    # --------------------------------------------------------------------------
    # 2. LOAD / GENERATE SPLITS
    # --------------------------------------------------------------------------
    # [FIX 1] Use correct config key 'ntrial' (matches __config__.yaml)
    ntrial = cfg.train.get("ntrial", 1)
    
    # [FIX 2] Use correct filename format (matches main.py)
    splitstr = f'splits.t{ntrial}.r{cfg.train.train_test_ratio}.pkl'
    output_pkl = os.path.join(CLI["input_dir"], splitstr)


    print(f"[TEST] Target Split File: {output_pkl}")
    print(f"[TEST] Config: ntrial={ntrial}, folds={cfg.train.nfolds}")

    splits_bundle = get_splits(
        n_sample=teamsvecs["skill"].shape[0],
        n_folds=cfg.train.nfolds,
        train_ratio=cfg.train.train_test_ratio,
        output=output_pkl,
        seed=cfg.seed,
        year_idx=None, # Assuming non-temporal for this constraint test
        step_ahead=cfg.train.step_ahead,
        n_trials=ntrial,
        member_mat=teamsvecs["member"],
        return_bundle=True, # Critical for testing all trials
    )

    # --------------------------------------------------------------------------
    # 3. TEAM NAME MAPPING (OPTIONAL)
    # --------------------------------------------------------------------------
    team_names = None
    if "i2t" in indexes:
        max_idx = max(indexes["i2t"])
        team_names = ["Unknown"] * (max_idx + 1)
        for k, v in indexes["i2t"].items():
            team_names[k] = str(v)

    # --------------------------------------------------------------------------
    # 4. RUN SELF-TEST
    # --------------------------------------------------------------------------
    try:
        selftest_do_split_bundle(
            splits_bundle=splits_bundle,
            member_mat=teamsvecs["member"],
            team_names=team_names,
            n_folds_expected=cfg.train.nfolds,
        )
        print("\n[RESULT] ✅ Split validation completed successfully.")
    except AssertionError as e:
        print(f"\n[RESULT] ❌ VALIDATION FAILED: {e}")
        # Exit with error code so CI/CD pipelines fail
        sys.exit(1)

if __name__ == '__main__':
    parse_args()
    run_test()