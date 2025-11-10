# radgraph_runner.py
"""
Helper functions to load RadGraph model and run inference.
This module hides the RadGraph import logic and returns processed annotations.
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Lazy import to avoid overhead at module import time in some environments.
def load_radgraph_model(model_type: str = "modern-radgraph-xl"):
    """
    Load and return an instance of the RadGraph inference wrapper.
    model_type: the name used by the RadGraph package to load a prepackaged model.
                Default is "modern-radgraph-xl" (matches upstream examples).
    Returns the RadGraph object.
    """
    try:
        # Import here (heavy)
        from radgraph import RadGraph
    except Exception as e:
        logger.exception("Failed importing radgraph package. Did you install it?")
        raise

    # Instantiate (this may download weights)
    try:
        rg = RadGraph(model_type=model_type)
    except Exception as e:
        logger.exception("Failed to initialize RadGraph with model_type=%s", model_type)
        raise

    return rg


def annotate_reports(radgraph_model, reports: List[str]) -> List[Dict[str, Any]]:
    """
    Run radgraph_model on a list of textual reports.
    Returns a list of processed-annotation dicts (model outputs) as produced by
    radgraph.get_radgraph_processed_annotations or the RadGraph wrapper.
    """
    try:
        # radgraph object typically supports calling with a list of texts
        raw_outputs = radgraph_model(reports)
        # The RadGraph wrapper may return processed annotations already, but to be safe:
        # try to convert using helper if available
        try:
            from radgraph import get_radgraph_processed_annotations
            # If the function exists, use it to normalize outputs
            processed = []
            for i, ro in enumerate(raw_outputs):
                # If input was a single string, radgraph may return mapping; handle both
                processed.append(get_radgraph_processed_annotations(ro))
            return processed
        except Exception:
            # If helper not available or returns unexpected format, return raw outputs
            return raw_outputs
    except Exception as e:
        logger.exception("Failed to run RadGraph model on input.")
        raise
