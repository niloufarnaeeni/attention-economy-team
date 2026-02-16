import os
import json
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# ARGPARSE
# -----------------------------
parser = argparse.ArgumentParser(description="Run GPT structured skill extraction on Web3 projects.")
parser.add_argument(
    "--csv",
    type=str,
    default=None,
    help="Path to the cleaned RootData CSV file. If not provided, default project CSV is used."
)
parser.add_argument(
    "--name-col",
    type=str,
    default="name",
    help="Which CSV column to use as project_name (default: name)"
)

parser.add_argument(
    "--out",
    type=str,
    default="default_run",
    help="Name of the output folder inside data/inference/"
)


args = parser.parse_args()


# -----------------------------
# PATHS & CONSTANTS
# -----------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent
DATA_DIR = _PROJECT_ROOT / "data"

# Default CSV path if argparse not provided
DEFAULT_CSV = DATA_DIR / "giverrep_with_rootdata_descriptions_cleaned.csv"

PROMPTS_FILE = DATA_DIR / "prompts_gpt.json"
OUTPUT_ROOT = DATA_DIR / "inference" / args.out
MODEL = "gpt-5-mini"

# Determine which CSV to use
CSV_FILE = Path(args.csv) if args.csv else DEFAULT_CSV
if not CSV_FILE.exists():
    raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")


# -----------------------------
# JSON Schema for structured output
# -----------------------------
WEB3_SCHEMA = {
    "type": "object",
    "properties": {
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "keyword": {"type": "string"},
                    "level": {"type": "string", "enum": ["Generic", "Specific"]},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 2,
                    },
                    "one_sentence_definition": {"type": "string"},
                },
                "required": [
                    "id",
                    "keyword",
                    "level",
                    "categories",
                    "one_sentence_definition",
                ],
                "additionalProperties": False,
            },
        },
        "projects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "project_name": {"type": "string"},
                    "assigned_skill_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "project_id",
                    "project_name",
                    "assigned_skill_ids",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["skills", "projects"],
    "additionalProperties": False,
}

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Missing OPENAI_API_KEY")

client = OpenAI()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Load prompts
with PROMPTS_FILE.open("r", encoding="utf-8") as f:
    prompts_cfg = json.load(f)

SYSTEM_PROMPT = prompts_cfg["system"]
USER_TEMPLATE = prompts_cfg["user_template"]

# Load project CSV (now from argparse)
df = pd.read_csv(CSV_FILE)

projects = []
for idx, row in df.iterrows():

    desc = str(row.get("root_enIntd", "")).strip()
    brief = str(row.get("root_enBriefIntd", "")).strip()
    tags = str(row.get("root_enTagNames", "")).strip()

    final_desc = f"""Description: {desc}
Brief: {brief}
Tags: {tags}"""
    
    project_name_col = args.name_col
    if project_name_col not in df.columns:
        raise KeyError(f"Column '{project_name_col}' not found in CSV. Available: {list(df.columns)}")
    projects.append(
        {
            "project_id": int(idx),
            "project_name": str(row[project_name_col]),
            "project_desc": final_desc,
        }
    )

# Build prompt
projects_json_str = json.dumps(projects, ensure_ascii=False, indent=2)
user_prompt = USER_TEMPLATE.replace("{{PROJECTS_JSON}}", projects_json_str)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_prompt},
]

# -----------------------------
# Call Responses API with structured output
# -----------------------------
response = client.responses.create(
    model=MODEL,
    input=messages,
    text={
        "format": {
            "type": "json_schema",
            "name": "web3_skill_schema",
            "strict": True,
            "schema": WEB3_SCHEMA,
        }
    },
)

# Save result
out_path = OUTPUT_ROOT / "gpt5_response_full.json"
with out_path.open("w", encoding="utf-8") as f:
    json.dump(response.model_dump(), f, ensure_ascii=False, indent=2)

print(f"Saved full GPT-5 response to {out_path}")
print(f"[INFO] Processed CSV: {CSV_FILE}")
