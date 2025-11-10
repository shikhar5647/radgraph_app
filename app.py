import streamlit as st
import json
import pandas as pd
from typing import List, Dict, Any
from radgraph_runner import load_radgraph_model, annotate_reports
import traceback

st.set_page_config(page_title="RadGraph Demo", layout="wide")

st.title("RadGraph — Report → Structured JSON")
st.markdown(
    """
Enter a radiology report in the left pane and press **Annotate**.
The app will run the RadGraph inference wrapper and show:
* Raw RadGraph JSON
* A parsed list of entities (Anatomy & Observation) and relation table
* Download buttons to save JSON/CSV/Excel
\n
**Important:** Do *not* paste real patient-identifying information (PHI) here unless you are cleared to do so.
"""
)

# Sidebar: model config
with st.sidebar:
    st.header("Model / settings")
    model_type = st.text_input("RadGraph model_type (artifact ID)", "modern-radgraph-xl")
    st.markdown(
        """
By default we use `modern-radgraph-xl`. If you have a different model artifact (e.g. hosted as `stanfordaimi/modern-radgraph-xl`),
enter that ID here. The RadGraph wrapper will attempt to load the selected artifact.
"""
    )
    show_raw = st.checkbox("Show raw model output", value=True)
    st.caption("First load may take time while the model files are downloaded.")

# Main layout: two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input report")
    default_text = (
        "Exam: Chest radiograph. Findings: The heart size is within normal limits. "
        "There is a small right pleural effusion. A 1.2 cm rounded density in the right lower lobe may represent a small neoplasm."
    )
    report_text = st.text_area("Paste radiology report here", value=default_text, height=300)
    annotate_btn = st.button("Annotate")

# Prepare placeholders
json_placeholder = st.empty()
entities_placeholder = st.empty()
relations_placeholder = st.empty()
download_placeholder = st.empty()
status_placeholder = st.empty()

# Cached model loader (Streamlit cache_resource for long-lived objects)
@st.cache_resource(show_spinner=False)
def _get_model(model_type_param: str):
    return load_radgraph_model(model_type=model_type_param)

if annotate_btn:
    if not report_text.strip():
        st.warning("Please enter a report to annotate.")
    else:
        try:
            status_placeholder.info("Loading model (if not cached) and running inference...")

            # Load or get cached model
            radgraph_model = _get_model(model_type)

            # Run inference (single-element list)
            outputs = annotate_reports(radgraph_model, [report_text])
            # outputs is a list (one per report)
            model_output = outputs[0]

            # If radgraph returns a nested object from get_radgraph_processed_annotations,
            # it may have keys like 'entities', 'relations' etc. We'll attempt to handle common shapes.
            # Show raw if requested
            if show_raw:
                st.subheader("Raw model output (JSON)")
                json_placeholder.json(model_output)

            # Extract entity list (normalize to list of dicts if needed)
            # Two common output shapes:
            # 1) {'entities': {id: {...}}, 'relations': [...]}  <-- processed_annotations
            # 2) a list/dict of spans in another format. We try the first, then fallback.

            def _extract_entities_relations(out) -> (List[Dict], List[Dict]):
                ents = []
                rels = []
                if isinstance(out, dict):
                    # case: processed annotation format
                    if 'entities' in out and isinstance(out['entities'], dict):
                        for eid, ed in out['entities'].items():
                            # typical fields: 'text' or 'tokens', 'label', 'start', 'end'
                            text = ed.get('text') or ed.get('tokens') or ed.get('tokens_text') or ""
                            label = ed.get('label') or ""
                            start = ed.get('start', None)
                            end = ed.get('end', None)
                            ents.append({
                                'id': eid,
                                'text': text,
                                'label': label,
                                'start': start,
                                'end': end
                            })
                    # relations: list of dicts or tuples
                    if 'relations' in out and isinstance(out['relations'], list):
                        for r in out['relations']:
                            # try common keys
                            if isinstance(r, dict):
                                src = r.get('source') or r.get('from') or r.get('head') or r.get('arg1')
                                tgt = r.get('target') or r.get('to') or r.get('tail') or r.get('arg2')
                                label = r.get('label') or r.get('type') or ""
                                rels.append({'source': src, 'target': tgt, 'label': label})
                            elif isinstance(r, (list, tuple)) and len(r) >= 3:
                                # maybe [src, tgt, label]
                                rels.append({'source': r[0], 'target': r[1], 'label': r[2]})
                # fallback: if out is a string or list
                return ents, rels

            entities, relations = _extract_entities_relations(model_output)

            # Display entities as a DataFrame
            if entities:
                df_ents = pd.DataFrame(entities)
                st.subheader("Extracted Entities")
                entities_placeholder.dataframe(df_ents)

                # Make CSV download
                csv_bytes = df_ents.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Entities CSV",
                    data=csv_bytes,
                    file_name="radgraph_entities.csv",
                    mime="text/csv",
                    key="download_csv"
                )

            else:
                st.info("No entities extracted in expected format. See raw JSON (enable 'Show raw model output').")

            # Display relations table
            if relations:
                df_rels = pd.DataFrame(relations)
                st.subheader("Extracted Relations")
                relations_placeholder.dataframe(df_rels)
            else:
                st.info("No relations found in the processed output.")

            # Always provide the raw JSON as a downloadable file
            json_blob = json.dumps(model_output, indent=2)
            st.download_button(
                label="Download JSON output",
                data=json_blob,
                file_name="radgraph_output.json",
                mime="application/json",
                key="download_json"
            )

            status_placeholder.success("Done — model ran successfully.")
        except Exception as e:
            status_placeholder.error("Error running RadGraph model. See details below.")
            st.error(traceback.format_exc())
