"""
Streamlit app to annotate a single radiology report using RadGraph loaded from Hugging Face.

Usage:
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import json
import os
import traceback

from radgraph_runner import load_radgraph_model, annotate_reports

st.set_page_config(page_title="RadGraph (HF) Demo", layout="wide")

st.title("RadGraph — HF-backed inference demo")

st.markdown(
    """
Enter a radiology report and press **Annotate**. The app will:
- authenticate to Hugging Face using `HUGGINGFACEHUB_API_TOKEN` from `.env` (or env),
- load the RadGraph model (first time may download using your token),
- return processed RadGraph JSON plus entity & relation tables.

**Warning:** Do not paste real PHI unless permitted by your policies.
"""
)

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    model_id = st.text_input("RadGraph HF model id / artifact alias", value=os.environ.get("RADGRAPH_MODEL_ID", "modern-radgraph-xl"))
    show_raw = st.checkbox("Show raw JSON output", value=True)
    st.write("Make sure your HUGGINGFACEHUB_API_TOKEN is set in environment or .env.")

# Input area
report_text = st.text_area("Radiology report text", height=300, value=(
    "Exam: Chest radiograph. Findings: The heart size is within normal limits. "
    "There is a small right pleural effusion. A 1.2 cm rounded density in the right lower lobe may represent a small neoplasm."
))

annotate_btn = st.button("Annotate")

# cached loader to keep model in memory across reruns
@st.cache_resource(show_spinner=False)
def get_model_cached(mid):
    return load_radgraph_model(model_id=mid)

if annotate_btn:
    if not report_text.strip():
        st.warning("Please enter a report.")
    else:
        try:
            st.info("Loading model (cached if previously loaded) and running inference...")

            model = get_model_cached(model_id)
            outputs = annotate_reports(model, [report_text])

            if not outputs:
                st.error("Model returned empty output. See logs / raw output.")
                st.stop()

            # pick first output (we passed a single report)
            model_output = outputs[0]

            if show_raw:
                st.subheader("Raw model output")
                st.json(model_output)

            # Attempt to extract entities and relations in common shapes
            entities = []
            relations = []

            if isinstance(model_output, dict):
                # common "processed annotations" format: 'entities' is a dict
                if 'entities' in model_output and isinstance(model_output['entities'], dict):
                    for eid, ent in model_output['entities'].items():
                        text = ent.get("text") or ent.get("tokens") or ent.get("tokens_text") or ent.get("tokens_text_joined") or ""
                        label = ent.get("label") or ""
                        start = ent.get("start")
                        end = ent.get("end")
                        entities.append({
                            "id": eid,
                            "text": text,
                            "label": label,
                            "start": start,
                            "end": end
                        })
                # fallback: maybe 'ner' + 'sentences' format
                elif 'ner' in model_output and 'sentences' in model_output:
                    # simple re-use of radgraph parsing conventions could be added here
                    pass

                # relations
                if 'relations' in model_output and isinstance(model_output['relations'], list):
                    for r in model_output['relations']:
                        if isinstance(r, dict):
                            src = r.get('source') or r.get('from') or r.get('head')
                            tgt = r.get('target') or r.get('to') or r.get('tail')
                            lbl = r.get('label') or r.get('type') or ""
                            relations.append({"source": src, "target": tgt, "label": lbl})
                        elif isinstance(r, (list, tuple)) and len(r) >= 3:
                            relations.append({"source": r[0], "target": r[1], "label": r[2]})

            # Display entities and relations
            if entities:
                st.subheader("Entities")
                df_ents = pd.DataFrame(entities)
                st.dataframe(df_ents)
                st.download_button("Download Entities CSV", df_ents.to_csv(index=False).encode('utf-8'),
                                   file_name="radgraph_entities.csv", mime="text/csv")
            else:
                st.info("No entities found in recognized format. Inspect raw JSON.")

            if relations:
                st.subheader("Relations")
                df_rels = pd.DataFrame(relations)
                st.dataframe(df_rels)
                st.download_button("Download Relations CSV", df_rels.to_csv(index=False).encode('utf-8'),
                                   file_name="radgraph_relations.csv", mime="text/csv")

            # Always provide full JSON download
            st.download_button("Download Raw JSON", json.dumps(model_output, indent=2).encode('utf-8'),
                               file_name="radgraph_output.json", mime="application/json")

            st.success("Annotation complete.")
        except Exception as e:
            st.error("Error during annotation. See traceback below.")
            st.text(traceback.format_exc())
