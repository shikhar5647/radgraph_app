"""
radgraph_runner.py

Utility to authenticate to HF, load the RadGraph model (from a HF model repo id or the upstream alias),
and run inference while normalizing output shapes.

This module intentionally keeps imports lazy to avoid heavy startup cost until needed.
"""
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  # optional dependency; OK if not available

from huggingface_hub import login as hf_login

def ensure_hf_auth():
    """
    Ensure huggingface_hub is authenticated either via HUGGINGFACEHUB_API_TOKEN env var
    or via existing login. This helps transformers/from_pretrained download private artifacts.
    """
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        logger.info("No HUGGINGFACEHUB_API_TOKEN found in environment; continuing anonymous (may fail for private repos).")
        return False
    try:
        hf_login(token=token, add_to_git_credential=False)
        logger.info("Logged in to Hugging Face hub using token.")
        return True
    except Exception as e:
        logger.exception("Failed to login to Hugging Face hub with provided token: %s", e)
        return False

def load_radgraph_model(model_id: str = None):
    """
    Load RadGraph inference wrapper. model_id can be:
      - None (we pick default from env RADGRAPH_MODEL_ID or 'modern-radgraph-xl')
      - A HF repo id or artifact alias supported by radgraph package.
    Returns an instantiated radgraph RadGraph wrapper.
    """
    ensure_hf_auth()

    if model_id is None:
        model_id = os.environ.get("RADGRAPH_MODEL_ID", "modern-radgraph-xl")
    logger.info("Loading RadGraph model with model_id=%s", model_id)

    try:
        # radgraph is a heavy import; do it lazily
        from radgraph import RadGraph
    except Exception as e:
        logger.exception("Failed to import radgraph package. Make sure 'radgraph' is installed.")
        raise

    try:
        rg = RadGraph(model_type=model_id)
    except Exception as e:
        logger.exception("RadGraph initialization failed for model_type=%s. Error: %s", model_id, e)
        raise

    logger.info("RadGraph model loaded.")
    return rg


def normalize_radgraph_outputs(raw_outputs) -> List[Dict[str, Any]]:
    """
    Normalize output to a list of per-document processed-annotation dicts.
    Many radgraph wrapper variants can return:
      - a dict (single doc)
      - a list of dicts (one per doc)
      - a tuple (maybe (model_obj, [annotations]))
    This function tries to produce a list of dicts and to call get_radgraph_processed_annotations
    if it exists.
    """
    # If single dict, wrap
    if isinstance(raw_outputs, dict):
        raw_outputs = [raw_outputs]

    # Attempt to recover typical tuple shapes
    if isinstance(raw_outputs, tuple) and len(raw_outputs) == 2 and isinstance(raw_outputs[1], (list, tuple)):
        raw_outputs = list(raw_outputs[1])

    # If it looks like (model_obj, annotations) sometimes both are present
    if isinstance(raw_outputs, list) and len(raw_outputs) == 2 and not isinstance(raw_outputs[0], dict) and isinstance(raw_outputs[1], list):
        raw_outputs = list(raw_outputs[1])

    processed = []
    for ro in raw_outputs:
        # try to transform via helper if available
        try:
            from radgraph import get_radgraph_processed_annotations
            try:
                pr = get_radgraph_processed_annotations(ro)
                processed.append(pr)
                continue
            except Exception:
                # helper may raise if ro already processed or is not expected shape
                pass
        except Exception:
            # helper not installed / available
            pass

        # fallback: ensure it's a dict (if not, wrap)
        if isinstance(ro, dict):
            processed.append(ro)
        else:
            # best-effort: put in a dict wrapper
            processed.append({"raw": ro})
    return processed

def annotate_reports(radgraph_model, reports: List[str]) -> List[Dict[str, Any]]:
    """
    Run the radgraph_model on a list of textual reports and return a list of processed dicts.
    This function handles wrappers that accept either a list or single string.
    """
    # Prefer to call with list, but be tolerant
    try:
        raw_outputs = radgraph_model(reports)
    except Exception as e:
        logger.warning("Calling radgraph with list failed: %s. Falling back to per-string calls.", e)
        raw_outputs = []
        for r in reports:
            try:
                raw_outputs.append(radgraph_model(r))
            except Exception as e2:
                logger.exception("radgraph failed on a single report: %s", e2)
                raw_outputs.append({"error": str(e2), "input": r})

    normalized = normalize_radgraph_outputs(raw_outputs)
    return normalized
