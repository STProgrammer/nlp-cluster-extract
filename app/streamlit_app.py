"""Streamlit UI for clustering + extraction demo."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from nlp_cluster_extract.clustering import run_clustering
from nlp_cluster_extract.extraction import run_extraction
from ui_helpers import (
    compute_stats,
    ensure_dir,
    filter_nonempty,
    load_extraction_json,
    preview_table,
    render_pattern_expanders,
    save_uploaded_csv,
    show_download_button,
    timestamp_label,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_news_blurbs.csv"
OUTPUT_ROOT = ROOT / "outputs" / "ui_runs"


st.set_page_config(page_title="NLP News Blurb Toolkit", layout="wide")
st.title("NLP News Blurb Toolkit: Clustering + Extraction")
st.write(
    "This demo clusters synthetic news blurbs and extracts keywords + patterns. "
    "All text is synthetic and local-first."
)

# ---- Session state ----
if "df" not in st.session_state:
    st.session_state.df = None
if "data_path" not in st.session_state:
    st.session_state.data_path = None
if "cluster_out" not in st.session_state:
    st.session_state.cluster_out = None
if "clusters_data" not in st.session_state:
    st.session_state.clusters_data = None
if "clustered_df" not in st.session_state:
    st.session_state.clustered_df = None
if "selected_cluster" not in st.session_state:
    st.session_state.selected_cluster = None
if "extract_out" not in st.session_state:
    st.session_state.extract_out = None

# ---- Data input panel ----
st.header("Data input")
use_sample = st.toggle("Use sample dataset", value=True)

uploaded_file = None
if not use_sample:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if use_sample:
    if not DATA_PATH.exists():
        st.error("Sample data not found. Run `nlpce sample-data` first.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    st.session_state.data_path = DATA_PATH
else:
    if uploaded_file is None:
        st.info("Upload a CSV to continue.")
        st.stop()
    tmp_dir = ensure_dir(OUTPUT_ROOT / "uploads")
    tmp_path = tmp_dir / uploaded_file.name
    save_uploaded_csv(uploaded_file, tmp_path)
    df = pd.read_csv(tmp_path)
    st.session_state.data_path = tmp_path

st.session_state.df = df

columns = list(df.columns)
default_text_col = "blurb" if "blurb" in columns else ("text" if "text" in columns else columns[0])
default_title_col = "headline" if "headline" in columns else None

text_col = st.selectbox("Text column", options=columns, index=columns.index(default_text_col))
if default_title_col and default_title_col in columns:
    title_default_idx = columns.index(default_title_col)
else:
    title_default_idx = 0

title_col_choice = st.selectbox(
    "Title column (optional)",
    options=["(none)"] + columns,
    index=1 if default_title_col else 0,
)

selected_cols = [text_col]
if title_col_choice != "(none)":
    selected_cols.insert(0, title_col_choice)

st.subheader("Preview")
st.dataframe(preview_table(df, selected_cols), width="stretch")

stats = compute_stats(df, text_col)
stat_cols = st.columns(3)
stat_cols[0].metric("Rows", stats.rows)
stat_cols[1].metric("Missing texts", stats.missing_texts)
stat_cols[2].metric("Avg length", stats.avg_length)

# ---- Clustering section ----
st.header("Clustering")

k = st.slider("k (clusters)", min_value=2, max_value=12, value=8, step=1)
col1, col2, col3 = st.columns(3)
with col1:
    top_terms = st.slider("Top terms", min_value=5, max_value=20, value=12)
with col2:
    min_df = st.number_input("min_df", min_value=1, max_value=10, value=2)
with col3:
    max_df = st.slider("max_df", min_value=0.5, max_value=1.0, value=0.9, step=0.05)

ngram_label = st.selectbox("ngram_range", options=["(1,1)", "(1,2)"], index=1)
ngram_range = (1, 1) if ngram_label == "(1,1)" else (1, 2)

if st.button("Run clustering"):
    out_dir = ensure_dir(OUTPUT_ROOT / f"{timestamp_label()}_cluster")
    title_col = None if title_col_choice == "(none)" else title_col_choice
    cleaned_df = filter_nonempty(df, text_col)
    if len(cleaned_df) < len(df):
        st.info("Empty texts were removed before clustering.")
    input_path = out_dir / "input_clean.csv"
    cleaned_df.to_csv(input_path, index=False)
    run_clustering(
        input_path=input_path,
        input_folder=None,
        out_dir=out_dir,
        text_col=text_col,
        title_col=title_col,
        id_col="item_id" if "item_id" in df.columns else "doc_id",
        k=int(k),
        min_df=int(min_df),
        max_df=float(max_df),
        ngram_range=ngram_range,
        top_n=int(top_terms),
    )
    st.session_state.cluster_out = out_dir
    clustered_path = out_dir / "clustered_docs.csv"
    clusters_path = out_dir / "clusters.json"
    if clustered_path.exists():
        st.session_state.clustered_df = pd.read_csv(clustered_path)
    if clusters_path.exists():
        st.session_state.clusters_data = json.loads(clusters_path.read_text(encoding="utf-8"))

cluster_out = st.session_state.cluster_out
clusters_data = st.session_state.clusters_data
clustered_df = st.session_state.clustered_df

if clusters_data:
    st.subheader("Clustering results")
    silhouette = clusters_data.get("metrics", {}).get("silhouette_score")
    st.metric("Silhouette score", f"{silhouette:.3f}" if silhouette is not None else "n/a")

    sizes = clusters_data.get("cluster_sizes", {})
    if sizes:
        size_series = pd.Series(sizes).sort_index()
        st.bar_chart(size_series)

    top_terms_table = []
    for cluster_id, terms in clusters_data.get("top_terms", {}).items():
        top_terms_table.append({"cluster": int(cluster_id), "top_terms": ", ".join(terms)})
    if top_terms_table:
        st.dataframe(pd.DataFrame(top_terms_table).sort_values("cluster"), width="stretch")

    if clusters_data.get("cluster_purity"):
        purity_rows = []
        for cluster_id, info in clusters_data["cluster_purity"].items():
            purity_rows.append(
                {
                    "cluster": int(cluster_id),
                    "top_topic": info.get("top_topic"),
                    "percent": info.get("percent"),
                }
            )
        st.subheader("Cluster purity")
        st.dataframe(pd.DataFrame(purity_rows).sort_values("cluster"), width="stretch")

    if clustered_df is not None and "cluster_id" in clustered_df.columns:
        cluster_ids = sorted(clustered_df["cluster_id"].unique().tolist())
        st.session_state.selected_cluster = st.selectbox(
            "View sample blurbs from cluster", options=cluster_ids, index=0
        )
        selection = clustered_df[clustered_df["cluster_id"] == st.session_state.selected_cluster]
        display_cols = [col for col in ["item_id", "headline", "snippet", "cluster_id"] if col in selection.columns]
        st.dataframe(selection[display_cols].head(10), width="stretch")

    if cluster_out:
        st.subheader("Download clustering outputs")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            show_download_button("clustered_docs.csv", cluster_out / "clustered_docs.csv", "text/csv")
        with col_b:
            show_download_button("clusters.json", cluster_out / "clusters.json", "application/json")
        with col_c:
            show_download_button("report.md", cluster_out / "report.md", "text/markdown")

# ---- Extraction section ----
st.header("Extraction")

keywords_top_k = st.slider("Top keywords per doc", min_value=5, max_value=20, value=12)
use_ner = st.checkbox("Enable spaCy NER if installed", value=False)

col_a, col_b = st.columns(2)
with col_a:
    run_all = st.button("Extract from ALL docs")
with col_b:
    run_cluster = st.button(
        "Extract only from selected cluster",
        disabled=clustered_df is None or st.session_state.selected_cluster is None,
    )

if run_all or run_cluster:
    out_dir = ensure_dir(OUTPUT_ROOT / f"{timestamp_label()}_extract")
    title_col = None if title_col_choice == "(none)" else title_col_choice

    if run_cluster and clustered_df is not None:
        id_column = "item_id" if "item_id" in clustered_df.columns else "doc_id"
        if id_column not in df.columns:
            st.warning("ID column not found in input data; extracting from all docs instead.")
            cleaned_df = filter_nonempty(df, text_col)
            input_path = out_dir / "input_clean.csv"
            cleaned_df.to_csv(input_path, index=False)
        else:
            selected_ids = clustered_df.loc[
                clustered_df["cluster_id"] == st.session_state.selected_cluster, id_column
            ].tolist()
            subset = df[df[id_column].isin(selected_ids)]
            subset = filter_nonempty(subset, text_col)
            temp_csv = out_dir / "cluster_subset.csv"
            subset.to_csv(temp_csv, index=False)
            input_path = temp_csv
    else:
        cleaned_df = filter_nonempty(df, text_col)
        if len(cleaned_df) < len(df):
            st.info("Empty texts were removed before extraction.")
        input_path = out_dir / "input_clean.csv"
        cleaned_df.to_csv(input_path, index=False)

    run_extraction(
        input_path=input_path,
        input_folder=None,
        out_dir=out_dir,
        text_col=text_col,
        title_col=title_col,
        id_col="item_id" if "item_id" in df.columns else "doc_id",
        top_k=int(keywords_top_k),
        use_ner=use_ner,
    )

    st.session_state.extract_out = out_dir

extract_out = st.session_state.extract_out
if extract_out:
    extraction_path = extract_out / "extraction.json"
    summary_path = extract_out / "keywords_summary.md"

    st.subheader("Extraction results")
    extraction_rows = load_extraction_json(extraction_path)
    if extraction_rows:
        table_rows = []
        for row in extraction_rows:
            table_rows.append(
                {
                    "doc_id": row.get("doc_id"),
                    "top_keywords": ", ".join(row.get("keywords", [])[:8]),
                }
            )
        st.dataframe(pd.DataFrame(table_rows), width="stretch")
        render_pattern_expanders(extraction_rows)

        if any("entities" in row for row in extraction_rows):
            label_counts: dict[str, int] = {}
            for row in extraction_rows:
                for ent in row.get("entities", []):
                    label = ent.get("label")
                    if label:
                        label_counts[label] = label_counts.get(label, 0) + 1
            st.subheader("NER entities by label")
            st.bar_chart(pd.Series(label_counts).sort_index())
        elif use_ner:
            st.info("spaCy model not installed; skipping NER.")
    else:
        st.info("No extraction results available yet.")

    st.subheader("Download extraction outputs")
    col_a, col_b = st.columns(2)
    with col_a:
        show_download_button("extraction.json", extraction_path, "application/json")
    with col_b:
        show_download_button("keywords_summary.md", summary_path, "text/markdown")
