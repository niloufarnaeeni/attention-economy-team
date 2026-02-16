import importlib
import os
import re


def infer_template_version_from_path(jsonl_path: str) -> str:
    name = os.path.basename(jsonl_path)
    m = re.search(r"_([vV]\d+)\.jsonl$", name)
    if not m:
        raise ValueError(f"Cannot infer template version from {name}")
    return m.group(1)


def load_template_metadata(template_version: str):
    """
    Load template metadata from templates.<template_version>
    """
    module_path = f"templates.{template_version}"
    module = importlib.import_module(module_path)

    return {
        "template_name": template_version,
        "has_neg": bool(getattr(module, "HAS_NEG", True)),
        "has_score": bool(getattr(module, "HAS_SCORE", False)),
    }


def load_templates(template_version: str):
    """
    Load QUERY_TEMPLATE and DOC_TEMPLATE from templates.<template_version>
    """
    module_path = f"templates.{template_version}"
    module = importlib.import_module(module_path)

    if not hasattr(module, "QUERY_TEMPLATE") or not hasattr(module, "DOC_TEMPLATE"):
        raise ValueError(
            f"{module_path} must define QUERY_TEMPLATE and DOC_TEMPLATE"
        )

    return module.QUERY_TEMPLATE, module.DOC_TEMPLATE
