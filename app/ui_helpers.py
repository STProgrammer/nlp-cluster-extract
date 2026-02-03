"""Helper utilities for the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import io

import pandas as pd
import streamlit as st


@dataclass
class DataStats:
    rows: int
    missing_texts: int
    avg_length: float


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_uploaded_csv(uploaded_file, dest: Path) -> Path:
    data = uploaded_file.read()
    dest.write_bytes(data)
    return dest


def preview_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return df.head(15)
    return df[existing].head(15)


def compute_stats(df: pd.DataFrame, text_col: str) -> DataStats:
    if text_col not in df.columns:
        return DataStats(rows=len(df), missing_texts=len(df), avg_length=0.0)
    texts = df[text_col].fillna("").astype(str)
    missing = (texts.str.strip() == "").sum()
    lengths = texts.str.len()
    avg_len = float(lengths.mean()) if len(lengths) else 0.0
    return DataStats(rows=len(df), missing_texts=int(missing), avg_length=round(avg_len, 2))


def filter_nonempty(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    if text_col not in df.columns:
        return df
    texts = df[text_col].fillna("").astype(str).str.strip()
    return df[texts != ""]


def show_download_button(label: str, path: Path, mime: str) -> None:
    if not path.exists():
        st.warning(f"Missing file: {path.name}")
        return
    data = path.read_bytes()
    st.download_button(label, data=data, file_name=path.name, mime=mime)


def render_pattern_expanders(extraction_rows: list[dict]) -> None:
    for row in extraction_rows:
        doc_id = row.get("doc_id")
        keywords = ", ".join(row.get("keywords", [])[:10])
        with st.expander(f"{doc_id} - {keywords}"):
            patterns = row.get("extracted_patterns", {})
            if any(patterns.get(key) for key in patterns):
                st.json(patterns)
            else:
                st.caption("No pattern matches found for this item.")
            if "entities" in row:
                st.json({"entities": row.get("entities", [])})


def load_extraction_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    return pd.read_json(io.StringIO(raw)).to_dict(orient="records")
