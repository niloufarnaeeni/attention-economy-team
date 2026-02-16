import os
import pickle
import logging
import numpy as np
import copy
import pandas as pd
from glob import glob
from functools import partial

import hydra
from omegaconf import OmegaConf 
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class
import pkgmgr as opentf
import scipy.sparse

log = logging.getLogger(__name__)

# ==============================================================================
# 1. HELPER: AGGREGATE RESULTS (AVERAGE OVER TRIALS)
# ==============================================================================
def aggregate_avg_over_trials(output):
    """
    Reads all test.pred.eval.mean.csv under output/** and computes:
    - mean_over_trials, std_over_trials for each (run_signature_without_trial, metric)
    """
    log.info(f"[AGG] Scanning for test.pred.eval.mean.csv under {output}")
    
    all_files = glob(os.path.join(output, "**", "test.pred.eval.mean.csv"), recursive=True)
    if not all_files:
        log.warning("[AGG] No evaluation files found.")
        return

    rows = []
    for fpath in all_files:
        rel = os.path.relpath(fpath, output)
        parts = rel.split(os.sep)

        # 1. Extract Trial ID from path (e.g., 'trial_0')
        trial_id = 0 # Default to 0 (Legacy/Single run)
        for p in parts:
            if p.startswith("trial_"):
                try:
                    trial_id = int(p.split("_", 1)[1])
                except Exception:
                    pass 
        
        # 2. Run Signature (remove trial folder)
        parts_wo_trial = [p for p in parts if not p.startswith("trial_")]
        run_sig = os.sep.join(parts_wo_trial[:-1])

        try:
            df = pd.read_csv(fpath, names=["metric", "mean", "std"], skiprows=1)
        except Exception as e:
            log.warning(f"[AGG] Failed to read {fpath}: {e}")
            continue

        # Force numeric
        df["mean"] = pd.to_numeric(df["mean"], errors='coerce')
        df["std"] = pd.to_numeric(df["std"], errors='coerce')

        df["trial_id"] = trial_id
        df["run_sig"] = run_sig
        rows.append(df)

    if not rows:
        return

    df_all = pd.concat(rows, ignore_index=True)
    df_all.dropna(subset=["mean"], inplace=True)

    # 3. Compute Mean & Std ACROSS trials
    g = df_all.groupby(["run_sig", "metric"], as_index=False)["mean"].agg(
        mean_over_trials="mean",
        std_over_trials="std",
        n_trials="count"
    )

    g["std_over_trials"] = g["std_over_trials"].fillna(0.0)

    out_csv = os.path.join(output, "test.pred.eval.mean.avg_over_trials.csv")
    g.to_csv(out_csv, index=False)
    log.info(f"[AGG] Saved averaged results to: {out_csv}")


# ==============================================================================
# 2. HELPER: CONSTRAINT SOLVER
# ==============================================================================
def _do_split_from_member_matrix(member_mat, train_ratio, seed, min_test_team_size=None, keep_ratio=True, max_swaps=20000):
    """
    Create ONE train/test split with the constraint:
      Every creator (member column) that appears in test must appear in train.
    """
    # [FIX] Ensure matrix is CSR so .getrow().indices works
    member_mat = member_mat.tocsr()

    project_sizes = np.diff(member_mat.indptr)
    
    if min_test_team_size is not None:
        eligible_for_test = np.where(project_sizes >= min_test_team_size)[0]
        print(f"[SPLIT] min_test_team_size = {min_test_team_size}")

    else:
        eligible_for_test = np.arange(member_mat.shape[0])

    rng = np.random.RandomState(seed)
    n_teams = member_mat.shape[0]
    eligible_for_test = np.asarray(eligible_for_test)
    eligible_set = set(eligible_for_test.tolist())


    n_test = int(round((1 - train_ratio) * n_teams))
    n_test = min(n_test, len(eligible_for_test))
    
    test_set = set(rng.choice(
        eligible_for_test,
        size=n_test,
        replace=False
    ).tolist())
    
    train_set = set(range(n_teams)) - test_set


    def col_counts(rows_set):
        if not rows_set:
            return np.zeros(member_mat.shape[1], dtype=np.int64)
        rows = np.fromiter(rows_set, dtype=int)
        # Summing a CSR matrix returns a numpy matrix, flatten to array
        return np.asarray(member_mat[rows].sum(axis=0)).ravel().astype(np.int64)

    train_cnt = col_counts(train_set)
    test_cnt  = col_counts(test_set)

    def bad_cols():
        return np.where((test_cnt > 0) & (train_cnt == 0))[0]

    swaps = 0
    bad = bad_cols()

    while bad.size > 0:
        c = int(bad[0])

        rows_with_c = member_mat[:, c].nonzero()[0]
        candidates = [r for r in rows_with_c if r in test_set]

        if not candidates:
            raise RuntimeError(
                f"Cannot fix split: creator-col {c} is in test_cnt but no test rows contain it. "
                "This indicates a logic / accounting error."
            )

        ti_in = candidates[rng.randint(len(candidates))]  # move test -> train

        test_set.remove(ti_in)
        train_set.add(ti_in)

        # This line was crashing. Now it will work because member_mat is CSR.
        cols_in = member_mat.getrow(ti_in).indices
        
        test_cnt[cols_in]  -= 1
        train_cnt[cols_in] += 1

        if keep_ratio:
            # only allow swap-out projects that are eligible for test
            train_list = list(train_set.intersection(eligible_set))
            rng.shuffle(train_list)


            swapped = False
            for ti_out in train_list:
                if swaps >= max_swaps:
                    break
                swaps += 1

                if ti_out == ti_in:
                    continue

                cols_out = member_mat.getrow(ti_out).indices

                if np.any(train_cnt[cols_out] <= 1):
                    continue

                train_set.remove(ti_out)
                test_set.add(ti_out)

                train_cnt[cols_out] -= 1
                test_cnt[cols_out]  += 1

                swapped = True
                break

            if not swapped:
                keep_ratio = False

        bad = bad_cols()
    
    # ==========================================================
    # [DEBUG ADDITION START]
    # ==========================================================
    n_train_final = len(train_set)
    n_test_final = len(test_set)
    total_final = n_train_final + n_test_final
    actual_ratio = n_train_final / total_final if total_final > 0 else 0.0

    print(f"\n" + "="*40)
    print(f"[DEBUG] SPLIT DIAGNOSTICS")
    print(f"========================================")
    print(f" Final Train Size: {n_train_final}")
    print(f" Final Test Size:  {n_test_final}")
    print(f" Final Ratio:      {actual_ratio:.4f} (Target was {train_ratio})")
    
    if not keep_ratio:
        print(f" NOTICE: The code GAVE UP on the ratio to ensure valid coverage.")

    
    print("="*40 + "\n")
    # ==========================================================
    # [DEBUG ADDITION END]
    # ==========================================================
    return np.array(sorted(train_set), dtype=int), np.array(sorted(test_set), dtype=int)

def _fixed_train_valid_split_keep_test_coverage(member_mat, train_pool, test_idx, valid_ratio=0.1, seed=0):
    """
    Make ONE (train, valid) split from train_pool, WITHOUT breaking the constraint:
      every creator that appears in test must appear in *train* (not only valid).

    Returns:
      train_sub (np.ndarray), valid (np.ndarray)
    """
    member_mat = member_mat.tocsr()

    train_pool = np.asarray(train_pool, dtype=int)
    test_idx   = np.asarray(test_idx, dtype=int)

    if valid_ratio <= 0 or train_pool.size == 0:
        return train_pool, np.array([], dtype=int)


    # If no creators in test, any split is fine (still deterministic)
    rng = np.random.RandomState(seed)
    pool = train_pool.copy()
    rng.shuffle(pool)

    target_valid = int(round(valid_ratio * pool.size))
    target_valid = max(1, min(target_valid, pool.size - 1))  # keep at least 1 train row

    # Count how many times each required creator appears in current train_sub
    train_sub_set = set(pool.tolist())
    # Count ALL creators in current train_sub (not only test creators)
    full_counts = np.asarray(member_mat[train_pool].sum(axis=0)).ravel().astype(np.int64)


    valid = []
    # Greedily move rows from train_sub -> valid if it doesn't remove the last required creator
    for r in pool:
        if len(valid) >= target_valid:
            break
        if len(train_sub_set) <= 1:
            break

        

        row_cols = member_mat.getrow(int(r)).indices
        

        # do NOT allow removing the last occurrence of ANY creator
        if np.any(full_counts[row_cols] <= 1):
            continue
        
        full_counts[row_cols] -= 1

        train_sub_set.remove(int(r))
        valid.append(int(r))

    train_sub = np.array(sorted(train_sub_set), dtype=int)
    valid = np.array(sorted(valid), dtype=int)

    return train_sub, valid


# ==============================================================================
# 3. HELPER: GET SPLITS
# ==============================================================================
def get_splits(n_sample, n_folds, train_ratio, output, seed, 
               year_idx=None, step_ahead=1, n_trials=5, trial_id=None, member_mat=None, min_test_team_size=None,
               return_bundle=False):
    scikit = opentf.install_import('scikit-learn', 'sklearn.model_selection')

    if os.path.exists(output):
        log.info(f'Loading splits from {output} ...')
        with open(output, 'rb') as f:
            loaded = pickle.load(f)

        is_bundle = isinstance(loaded, dict) and 'trials' in loaded

        if is_bundle:
            if return_bundle: return loaded
            if trial_id is not None: return loaded['trials'][int(trial_id)]
            return loaded['trials'][0] 

        if return_bundle: 
            return {'trials': {0: loaded}, 'n_trials': 1}
        
        if trial_id is not None and int(trial_id) > 0:
            raise ValueError(
                f"Requested trial_id={trial_id}, but found legacy single-split file at {output}. "
                f"Delete it to regenerate the trials bundle."
            )
        return loaded

    log.info('Splits file not found! Generating ...')

    if year_idx is not None:
        raise NotImplementedError("Temporal split is disabled for now. Pass year_idx=None.")
        valid_ratio = 0.1
        train_pool = train.copy()
        # deterministic: last X% as valid
        n_valid = int(round(valid_ratio * train_pool.size))
        n_valid = max(1, min(n_valid, train_pool.size - 1))
        valid = np.array(sorted(train_pool[-n_valid:]), dtype=int)
        train_sub = np.array(sorted(train_pool[:-n_valid]), dtype=int)
        
        splits = {'test': test, 'folds': {0: {'train': train_sub, 'valid': valid}}}

        with open(output, 'wb') as f:
            pickle.dump(splits, f)
        
        if return_bundle: return {'trials': {0: splits}, 'n_trials': 1}
        return splits

    trials = {}
    seen_tests = set()

    for t in range(n_trials):
        this_seed = int(seed) + (t * 1000)

        attempt = 0
        while True:
            if member_mat is not None:
                train, test = _do_split_from_member_matrix(member_mat, train_ratio, this_seed, min_test_team_size=min_test_team_size, keep_ratio=True)
            else:
                train, test = scikit.train_test_split(
                    np.arange(n_sample),
                    train_size=train_ratio,
                    random_state=this_seed,
                    shuffle=True
                )

            key = frozenset(test.tolist())
            if key not in seen_tests:
                seen_tests.add(key)
                break

            attempt += 1
            if attempt > 100:
                raise RuntimeError(f"Could not generate unique trial {t} after 100 attempts.")
            this_seed += 1

        split_t = {'test': np.array(test, dtype=int), 'folds': {}}
        # One fixed train/valid split per trial (no KFold)
        valid_ratio = 0.1  # you can later expose this in cfg if you want, but not required
        
        train_sub, valid = _fixed_train_valid_split_keep_test_coverage(
            member_mat=member_mat if member_mat is not None else scipy.sparse.csr_matrix((n_sample, 1)),
            train_pool=train,
            test_idx=test,
            valid_ratio=valid_ratio,
            seed=this_seed
        )
        
        split_t['folds'] = {
            0: {'train': train_sub, 'valid': valid}
        }


        trials[t] = split_t

    bundle = {'trials': trials, 'n_trials': n_trials}
    with open(output, 'wb') as f:
        pickle.dump(bundle, f)

    if return_bundle: return bundle
    if trial_id is not None: return trials[int(trial_id)]
    return trials[0]



# ==============================================================================
# 4. MAIN RUN FUNCTION
# ==============================================================================
@hydra.main(version_base=None, config_path='.', config_name='__config__')
def run(cfg):
    domain_cls = get_class(cfg.data.domain)

    cfg.data.output += (
        f'.mt{cfg.data.filter.min_nteam}.ts{cfg.data.filter.min_team_size}'
        if 'filter' in cfg.data and cfg.data.filter else ''
    )

    # Ensure output_dir exists
    os.makedirs(cfg.data.output, exist_ok=True)

    cache_dir = getattr(cfg.data, "cache_dir", None)
    if cache_dir is None:
        cache_dir = cfg.data.output

    # Ensure cache_dir exists (important if cache_dir != output_dir)
    os.makedirs(cache_dir, exist_ok=True) 
    
    # ==========================================================
    # BUILD TEAM VECS ONCE (source of truth)
    # ==========================================================
    teamsvecs_orig, indexes = domain_cls.gen_teamsvecs(
        cfg.data.source,
        cache_dir,   # cache goes here, not output
        cfg.data
    )

    # ==========================================================
    # APPLY ALL ROW-CHANGING OPERATIONS *BEFORE* SPLITTING
    # ==========================================================
    if cfg.train.merge_teams_w_same_skills:
        log.info("Merging teams with identical skill sets (PRE-SPLIT)")
        domain_cls.merge_teams_by_skills(teamsvecs_orig, inplace=True)
    
        assert teamsvecs_orig["skill"].shape[0] == teamsvecs_orig["member"].shape[0], \
        "[FATAL] skill/member row mismatch after merge"

    # ==========================================================
    # FREEZE DATASET SIZE (SPLITS DEPEND ON THIS)
    # ==========================================================
    N_TEAMS_FROZEN = teamsvecs_orig['member'].shape[0]
    log.info(f"[FREEZE] Number of teams frozen at {N_TEAMS_FROZEN}")

    # Handle Year Indexes (Temporal)
    #year_idx = []
    #for i in range(1, len(indexes['i2y'])): 
    #     if indexes['i2y'][i][0] - indexes['i2y'][i-1][0] > cfg.train.nfolds: year_idx.append(indexes['i2y'][i-1])
    #year_idx.append(indexes['i2y'][-1])
    #indexes['i2y'] = year_idx

    # ------------------------------------------------------------------
    # PREPARE CONFIGS ONCE (Optimization)
    # ------------------------------------------------------------------
    final_emb_cfg = None
    if 'embedding' in cfg.data and cfg.data.embedding.class_method:
         emb_overrides = [o.replace('+data.embedding.', '') for o in HydraConfig.get().overrides.task if '+data.embedding.' in o]
         base_emb_cfg = OmegaConf.load(cfg.data.embedding.config)
         final_emb_cfg = OmegaConf.merge(base_emb_cfg, OmegaConf.from_dotlist(emb_overrides))
         final_emb_cfg.model.spe = cfg.train.save_per_epoch
         OmegaConf.resolve(final_emb_cfg)

    final_mdl_cfg = None
    if cfg.cmd and any(c in cfg.cmd for c in ['train', 'test', 'eval']):
         mdl_overrides = [o.replace('+models.', '') for o in HydraConfig.get().overrides.task if '+models.' in o]
         base_mdl_cfg = OmegaConf.load(cfg.models.config)
         final_mdl_cfg = OmegaConf.merge(base_mdl_cfg, OmegaConf.from_dotlist(mdl_overrides))
         final_mdl_cfg.spe = cfg.train.save_per_epoch
         final_mdl_cfg.tntf.tfolds = cfg.train.nfolds
         final_mdl_cfg.tntf.step_ahead = cfg.train.step_ahead
         OmegaConf.resolve(final_mdl_cfg)
    # ------------------------------------------------------------------

    # 2. GET SPLITS (Filename now includes .t{ntrial})
    # This separates 'splits.f3.r0.85.t1.pkl' (legacy) from 'splits.f3.r0.85.t5.pkl' (bundle)
    ntrial = cfg.train.get('ntrial', 1)
    is_multi_trial = ntrial > 1
    
    split_filename = f"splits.t{ntrial}.r{cfg.train.train_test_ratio}.pkl"
    splits_path = os.path.join(cache_dir, split_filename)

    min_test_team_size = cfg.test.get("min_test_team_size", None)

    
    

    splits_bundle = get_splits(
            n_sample=N_TEAMS_FROZEN,
            n_folds=cfg.train.nfolds,
            train_ratio=cfg.train.train_test_ratio,
            output=splits_path,
            seed=cfg.seed,
            year_idx=None,
            step_ahead=cfg.train.step_ahead,
            n_trials=ntrial,
            trial_id=None,
            member_mat=teamsvecs_orig['member'],
            min_test_team_size=min_test_team_size,
            return_bundle=is_multi_trial
        )


    if isinstance(splits_bundle, dict) and 'trials' in splits_bundle:
        trials_map = splits_bundle['trials']
    else:
        trials_map = {0: splits_bundle}

    # 3. LOOP TRIALS
    for tid, current_split in trials_map.items():
        # ==========================================================
        # ASSERTION GUARD — SPLITS MUST MATCH FROZEN DATASET
        # ==========================================================
        n_rows_now = teamsvecs_orig["member"].shape[0]
        assert n_rows_now == N_TEAMS_FROZEN, (
            f"[FATAL] Dataset row count changed after splitting. "
            f"Expected {N_TEAMS_FROZEN}, got {n_rows_now}. "
            f"Split indices are invalid."
        )

        if len(current_split["test"]) > 0:
            max_idx = int(np.max(current_split["test"]))
            assert max_idx < N_TEAMS_FROZEN, (
                f"[FATAL] Split index {max_idx} out of range "
                f"(n_teams={N_TEAMS_FROZEN}). "
                f"Likely loading incompatible cached splits."
            )

        
        trial_seed = int(cfg.seed) + (int(tid) * 1000)

        if len(trials_map) > 1:
            effective_output = os.path.join(cfg.data.output, f"trial_{tid}")
            if not os.path.isdir(effective_output): os.makedirs(effective_output)
        else:
            effective_output = cfg.data.output

        log.info(f"--- Running Trial {tid} (Seed {trial_seed}) --- Output: {effective_output}")

        # State Isolation
        trial_teamsvecs = copy.deepcopy(teamsvecs_orig)

        # Cache Isolation (Skill Coverage)
        trial_teamsvecs['skillcoverage'] = domain_cls.gen_skill_coverage(
            trial_teamsvecs, 
            effective_output, 
            skipteams=current_split['test']
        )

        # --- Embedding Training ---
        t2v = None
        if final_emb_cfg:
             cls_name, method = cfg.data.embedding.class_method.split('_') if '_' in cfg.data.embedding.class_method else (cfg.data.embedding.class_method, None)
             cls = get_class(cls_name)
             t2v = cls(effective_output, cfg.acceleration, trial_seed, final_emb_cfg.model[cls.__name__.lower()], method)
             t2v.learn(trial_teamsvecs, current_split)

        # --- Model Training ---
        if final_mdl_cfg:
            #if cfg.train.merge_teams_w_same_skills: 
             #   domain_cls.merge_teams_by_skills(trial_teamsvecs, inplace=True)
            
            if t2v:
                skill_vecs = t2v.get_dense_vecs(trial_teamsvecs, vectype='skill')
                assert skill_vecs.shape[0] == trial_teamsvecs['skill'].shape[0], 'Incorrect number of embeddings!'
                
                # [RESTORED] Original metrics compatibility
                trial_teamsvecs['original_skill'] = trial_teamsvecs['skill']
                trial_teamsvecs['skill'] = skill_vecs

            models = {}

            for m in cfg.models.instances:
                cls_method = m.split('_')
                cls = get_class(cls_method[0])
                model_out = t2v.output if t2v else effective_output
                
                if cls_method[0] == 'mdl.emb.gnn.Gnn':
                    assert t2v, 'GNN needs embedding!'
                    models[m] = t2v
                else:
                    models[m] = cls(model_out, cfg.acceleration, trial_seed, final_mdl_cfg[cls.__name__.lower()])
                
                if len(cls_method) > 1:
                    inner = get_class(cls_method[1])
                    models[m].model = inner(model_out, cfg.acceleration, trial_seed, final_mdl_cfg[inner.__name__.lower()])

                if 'train' in cfg.cmd: 
                     if cls_method[0] != 'mdl.emb.gnn.Gnn': models[m].learn(trial_teamsvecs, current_split, None)
                
                if 'test' in cfg.cmd: 
                    models[m].test(trial_teamsvecs, current_split, cfg.test)
                
                if 'eval' in cfg.cmd: 
                    eval_cfg = copy.deepcopy(cfg.eval)
                    for key in eval_cfg.metrics: eval_cfg.metrics[key] = [x.replace('topk', cfg.eval.topk) for x in eval_cfg.metrics[key]]
                    models[m].evaluate(trial_teamsvecs, current_split, eval_cfg)

    # 4. Aggregate Results (Average over trials)
    log.info(f'{opentf.textcolor["green"]}Aggregating and averaging results across trials... {opentf.textcolor["reset"]}')
    aggregate_avg_over_trials(cfg.data.output)

if __name__ == '__main__': run()