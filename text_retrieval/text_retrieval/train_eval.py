import argparse
import subprocess
import yaml
import sys
import tempfile
from pathlib import Path
from itertools import product
from copy import deepcopy

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(cfg: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def infer_datasets_from_test_jsonl(test_jsonl: str):
    """
    Given:
      data/<base>/processed/test.jsonl

    Returns:
      train, valid, test dataset paths
    """
    test_path = Path(test_jsonl).resolve()

    processed_dir = test_path.parent
    base_dir = processed_dir.parent

    train_dataset = processed_dir / "train.jsonl"
    val_dataset = processed_dir / "valid.jsonl"
    test_dataset = processed_dir / "test.jsonl"

    for p in [train_dataset, val_dataset, test_dataset]:
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")

    return {
        "train_dataset": str(train_dataset),
        "val_dataset": str(val_dataset),
        "test_dataset": str(test_dataset),
        "base_folder": base_dir.name,
    }


def run_command(cmd):
    print("\n" + "=" * 90)
    print("RUNNING:")
    print(" ".join(cmd))
    print("=" * 90)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    print(proc.stdout)
    return proc.stdout

def extract_run_dir(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("RUN_DIR::"):
            return Path(line.replace("RUN_DIR::", "").strip())
    raise RuntimeError("RUN_DIR not found in training output")

# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_config", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--output_dir", default="output/kaito")
    parser.add_argument("--ks", default="2,5,10")

    # optional overrides
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--losses", nargs="*", default=None)
    parser.add_argument("--lambdas", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument(
    "--use_yaps",
    action="store_true",
    help="If set, enable Yaps for both training-aware evaluation and inference sweep"
)


    args = parser.parse_args()

    # -----------------------------
    # Defaults
    # -----------------------------
    models = args.models or [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "BAAI/bge-reranker-base",
    ]

    losses = args.losses or (
    # pure ranking losses
    [
        "pairwise_ranknet",
        "listwise_ce",
    ]
    # ranking + structural regularization
    #+ [
    #    "pairwise_ranknet_centered",
    #]
    # bias-aware (Yaps-aware) ranking losses
    + [
        "pairwise_ranknet_yap_neg",
        "pairwise_ranknet_yap_both",
        "pairwise_ranknet_yap_pos"
    ]
    )

    DATASET_TRAIN_GROUP_SIZE = {
    "giverep": 4,
    "kaito": 16,
    "cookie_fun": 8,
}

    lambdas = args.lambdas

    base_cfg = load_yaml(Path(args.base_config))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Experiment grid
    # -------------------------------------------------
    for model_name, loss_type in product(models, losses):

        # Only sweep lambda for losses that actually use it
        if loss_type in ["pairwise_ranknet_centered", "pairwise_ranknet_yap_neg", "pairwise_ranknet_yap_both", "pairwise_ranknet_yap_pos"]:
            lambda_values = deepcopy(lambdas)
        else:
            lambda_values = [0.0]

        for train_lambda in lambda_values:

            print("\n" + "#" * 100)
            print(f"MODEL={model_name}")
            print(f"LOSS={loss_type}")
            print(f"TRAIN_LAMBDA={train_lambda}")
            print("#" * 100)

            # -------------------------------------------------
            # 1. Build TEMP config (hard override)
            # -------------------------------------------------
            dataset_info = infer_datasets_from_test_jsonl(args.test_jsonl)
            print(f"[DATASET] base_folder = {dataset_info['base_folder']}")
            cfg = dict(base_cfg)
            
            cfg["train_dataset"] = dataset_info["train_dataset"]
            cfg["val_dataset"] = dataset_info["val_dataset"]
            cfg["test_dataset"] = dataset_info["test_dataset"]
            cfg["output_dir"] = str(output_dir)

            cfg["model_name_or_path"] = model_name
            cfg["loss_type"] = loss_type
            cfg["lambda"] = float(train_lambda)
            
            # -----------------------------
            # Dataset-specific group size
            # -----------------------------
            matched = False
            for key, group_size in DATASET_TRAIN_GROUP_SIZE.items():
                if key in dataset_info['base_folder']:
                    cfg["train_group_size"] = group_size
                    matched = True
                    break
            
            if not matched:
                raise ValueError(
                    f"Cannot infer train_group_size from dataset folder: {dataset_info['base_folder']}"
                )

            print(f"[INFO] Using train_group_size={cfg['train_group_size']} "
                f"for dataset={dataset_info['base_folder']}")

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
                save_yaml(cfg, tmp_path)

            print("\nUSING CONFIG:")
            print(yaml.safe_dump(cfg))

            # -------------------------------------------------
            # 2. TRAIN
            # -------------------------------------------------
            train_cmd = [
                "accelerate", "launch",
                "-m", "rag_retrieval.train.reranker.train_reranker",
                "--config", str(tmp_path),
            ]

            train_out = run_command(train_cmd)
            run_dir = extract_run_dir(train_out)
            model_dir = run_dir / "model"

            if not model_dir.exists():
                raise RuntimeError(f"Model dir missing: {model_dir}")

            # -------------------------------------------------
            # 3. EVAL — baseline (NO YAPS)
            # -------------------------------------------------
            eval_base = [
                sys.executable, "-m", "rag_retrieval.infer.eval.evaluate_reranker",
                "--jsonl", args.test_jsonl,
                "--model", str(model_dir),
                "--ks", args.ks,
                "--output_dir", str(output_dir),
            ]

            run_command(eval_base)

            # -------------------------------------------------
            # 4. EVAL — YAPS inference sweep (ONLY if enabled)
            # -------------------------------------------------
            if args.use_yaps:
                for infer_lambda in lambdas:
                    eval_yap = [
                        sys.executable, "-m", "rag_retrieval.infer.eval.evaluate_reranker",
                        "--jsonl", args.test_jsonl,
                        "--model", str(model_dir),
                        "--ks", args.ks,
                        "--output_dir", str(output_dir),
                        "--use_yaps",
                        "--yaps_lambda", str(infer_lambda),
                    ]
                    run_command(eval_yap)
            else:
                print("[INFO] Skipping Yaps evaluation (use_yaps=False)")

            # cleanup temp config
            tmp_path.unlink(missing_ok=True)

    print("\nALL EXPERIMENTS FINISHED SUCCESSFULLY")

if __name__ == "__main__":
    main()
