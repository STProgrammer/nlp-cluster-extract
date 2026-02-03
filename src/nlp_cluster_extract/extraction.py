"""Keyword + pattern extraction pipeline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable
import json
import re

import pandas as pd
from rich.console import Console
import yake

from nlp_cluster_extract.clustering import load_documents


console = Console()

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
URL_RE = re.compile(r"\bhttps?://[^\s)]+|\bwww\.[^\s)]+", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
    r"[a-z]*\s+\d{1,2},\s+\d{4})\b",
    re.IGNORECASE,
)
MONEY_RE = re.compile(r"\b(?:USD\s*)?\$?\d+(?:,\d{3})*(?:\.\d{2})?\b")


def extract_patterns(text: str) -> dict[str, list[str]]:
    """Extract common patterns via regex."""
    patterns = {
        "emails": EMAIL_RE.findall(text),
        "urls": URL_RE.findall(text),
        "phones": PHONE_RE.findall(text),
        "dates": DATE_RE.findall(text),
        "money": MONEY_RE.findall(text),
    }
    cleaned: dict[str, list[str]] = {}
    for key, values in patterns.items():
        deduped = []
        seen = set()
        for value in values:
            value = value.strip(".,;:()[]{}<>")
            if value and value not in seen:
                seen.add(value)
                deduped.append(value)
        cleaned[key] = deduped
    return cleaned


def _load_spacy_model() -> object | None:
    try:
        import spacy  # type: ignore
    except Exception:
        return None
    for model_name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
        try:
            return spacy.load(model_name)
        except Exception:
            continue
    return None


def extract_entities(text: str, nlp) -> list[dict[str, str]]:
    """Extract selected entities with spaCy."""
    keep = {"PERSON", "ORG", "GPE", "DATE", "MONEY"}
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in keep:
            entities.append({"text": ent.text, "label": ent.label_})
    return entities


def extract_keywords(texts: Iterable[str], top_k: int) -> list[list[str]]:
    """Extract YAKE keywords per document."""
    extractor = yake.KeywordExtractor(lan="en", n=3, top=top_k)
    results: list[list[str]] = []
    for text in texts:
        keywords = [kw for kw, _score in extractor.extract_keywords(text)]
        results.append(keywords)
    return results


def build_keywords_summary(
    doc_keywords: list[list[str]],
    top_k: int,
) -> tuple[list[tuple[str, int]], Counter]:
    counter = Counter()
    for keywords in doc_keywords:
        counter.update([kw.lower() for kw in keywords])
    return counter.most_common(top_k), counter


def run_extraction(
    input_path: Path | None,
    input_folder: Path | None,
    out_dir: Path,
    text_col: str = "text",
    title_col: str | None = "title",
    id_col: str = "doc_id",
    top_k: int = 12,
    use_ner: bool = False,
) -> list[dict]:
    """Run keyword + pattern extraction and write outputs."""
    console.print("[bold]Loading documents...[/bold]")
    df = load_documents(
        input_path=input_path,
        input_folder=input_folder,
        text_col=text_col,
        title_col=title_col,
        id_col=id_col,
    )

    texts = df["text"].astype(str).tolist()

    console.print("[bold]Extracting keywords with YAKE...[/bold]")
    doc_keywords = extract_keywords(texts, top_k=top_k)

    console.print("[bold]Extracting regex patterns...[/bold]")
    patterns = [extract_patterns(text) for text in texts]

    nlp = None
    if use_ner:
        console.print("[bold]Attempting spaCy NER...[/bold]")
        nlp = _load_spacy_model()
        if not nlp:
            console.print("[yellow]spaCy model not installed; skipping NER.[/yellow]")

    entities: list[list[dict[str, str]]] | None = None
    if use_ner and nlp:
        entities = [extract_entities(text, nlp) for text in texts]

    extraction_rows: list[dict] = []
    for idx, doc_id in enumerate(df["doc_id"].tolist()):
        row = {
            "doc_id": doc_id,
            "keywords": doc_keywords[idx],
            "extracted_patterns": patterns[idx],
        }
        if entities is not None:
            row["entities"] = entities[idx]
        extraction_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    extraction_path = out_dir / "extraction.json"
    extraction_path.write_text(json.dumps(extraction_rows, indent=2), encoding="utf-8")

    top_keywords, keyword_counts = build_keywords_summary(doc_keywords, top_k=top_k)
    summary_path = out_dir / "keywords_summary.md"
    summary_lines = [
        "# Keyword Summary",
        "",
        "## Top Keywords",
        "",
        "| Keyword | Count |",
        "| --- | --- |",
    ]
    for keyword, count in top_keywords:
        summary_lines.append(f"| {keyword} | {count} |")

    clustered_path = out_dir / "clustered_docs.csv"
    if clustered_path.exists():
        try:
            clustered_df = pd.read_csv(clustered_path)
            id_col = None
            if "doc_id" in clustered_df.columns:
                id_col = "doc_id"
            elif "item_id" in clustered_df.columns:
                id_col = "item_id"

            if id_col and "cluster_id" in clustered_df.columns:
                summary_lines.extend(["", "## Cluster Keyword Highlights", ""])
                merged = clustered_df[[id_col, "cluster_id"]].merge(
                    pd.DataFrame({"doc_id": df["doc_id"], "keywords": doc_keywords}),
                    left_on=id_col,
                    right_on="doc_id",
                    how="left",
                )
                for cluster_id, group in merged.groupby("cluster_id"):
                    cluster_counter = Counter()
                    for keywords in group["keywords"].dropna():
                        cluster_counter.update([kw.lower() for kw in keywords])
                    top_cluster = ", ".join(
                        [kw for kw, _count in cluster_counter.most_common(top_k)]
                    )
                    summary_lines.append(f"- Cluster {cluster_id}: {top_cluster}")
        except Exception:
            console.print("[yellow]Could not read clustered_docs.csv for aggregation.[/yellow]")

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    console.print(f"[green]Wrote outputs to[/] {out_dir}")
    return extraction_rows
