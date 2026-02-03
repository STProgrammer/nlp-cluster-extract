"""Clustering pipeline for document collections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

import pandas as pd
from rich.console import Console
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import numpy as np

from nlp_cluster_extract.io_utils import load_txt_folder
from nlp_cluster_extract.preprocess import basic_clean


console = Console()


@dataclass
class ClusterResults:
    clustered_docs: pd.DataFrame
    metrics: dict
    cluster_sizes: dict[int, int]
    top_terms: dict[int, list[str]]
    representative_doc_ids: dict[int, str]
    cluster_purity: dict[int, dict[str, object]] | None = None


def _load_csv(input_path: Path, text_col: str, title_col: str | None, id_col: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}' in {input_path}")

    df = df.copy()
    df["text"] = df[text_col].astype(str)

    if id_col in df.columns:
        df["doc_id"] = df[id_col].astype(str)
    elif "doc_id" in df.columns:
        df["doc_id"] = df["doc_id"].astype(str)
        id_col = "doc_id"
    elif "item_id" in df.columns:
        df["doc_id"] = df["item_id"].astype(str)
        id_col = "item_id"
    else:
        df["doc_id"] = [f"doc_{idx:04d}" for idx in range(1, len(df) + 1)]

    columns = ["doc_id", "text"]
    if title_col and title_col in df.columns:
        columns.insert(1, title_col)
    if id_col in df.columns and id_col not in columns:
        columns.insert(0, id_col)
    if "topic_true" in df.columns:
        columns.append("topic_true")
    return df[columns]


def load_documents(
    input_path: Path | None,
    input_folder: Path | None,
    text_col: str,
    title_col: str | None,
    id_col: str,
) -> pd.DataFrame:
    """Load documents from a CSV or folder of .txt files."""
    if input_path and input_folder:
        raise ValueError("Use either --input or --input-folder, not both.")
    if not input_path and not input_folder:
        raise ValueError("Provide --input CSV path or --input-folder with .txt files.")

    if input_folder:
        df = load_txt_folder(input_folder)
        return df

    if not input_path.exists():
        raise FileNotFoundError(f"CSV not found: {input_path}")

    return _load_csv(input_path, text_col=text_col, title_col=title_col, id_col=id_col)


def _make_snippet(text: str, max_len: int = 160) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _vectorize(
    texts: Iterable[str],
    min_df: float,
    max_df: float,
    ngram_range: tuple[int, int],
) -> tuple[TfidfVectorizer, object]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
    )
    matrix = vectorizer.fit_transform(texts)
    if matrix.shape[1] == 0:
        raise ValueError("No features extracted; adjust min_df/max_df or input text.")
    return vectorizer, matrix


def _cluster(matrix, k: int, random_state: int = 42, n_init: int = 10) -> KMeans:
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    model.fit(matrix)
    return model


def _compute_silhouette(matrix, labels: list[int], k: int) -> float | None:
    if k <= 1:
        return None
    if matrix.shape[0] <= k:
        return None
    if len(set(labels)) < 2:
        return None
    try:
        return float(silhouette_score(matrix, labels))
    except Exception:
        return None


def _top_terms(
    vectorizer: TfidfVectorizer,
    kmeans: KMeans,
    top_n: int,
) -> dict[int, list[str]]:
    terms = vectorizer.get_feature_names_out()
    top_terms: dict[int, list[str]] = {}
    for cluster_id, center in enumerate(kmeans.cluster_centers_):
        top_idx = center.argsort()[::-1][:top_n]
        top_terms[cluster_id] = [terms[i] for i in top_idx]
    return top_terms


def _representative_docs(kmeans: KMeans, matrix, doc_ids: list[str]) -> dict[int, str]:
    distances = kmeans.transform(matrix)
    representatives: dict[int, str] = {}
    for cluster_id in range(kmeans.n_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
        if cluster_indices.size == 0:
            continue
        local_best = distances[cluster_indices, cluster_id].argmin()
        absolute_index = int(cluster_indices[local_best])
        representatives[cluster_id] = doc_ids[absolute_index]
    return representatives


def run_clustering(
    input_path: Path | None,
    input_folder: Path | None,
    out_dir: Path,
    text_col: str = "text",
    title_col: str | None = "title",
    id_col: str = "doc_id",
    k: int = 6,
    min_df: float = 2,
    max_df: float = 0.9,
    ngram_range: tuple[int, int] = (1, 2),
    top_n: int = 8,
) -> ClusterResults:
    """Run the full clustering pipeline and write outputs."""
    console.print("[bold]Loading documents...[/bold]")
    df = load_documents(
        input_path=input_path,
        input_folder=input_folder,
        text_col=text_col,
        title_col=title_col,
        id_col=id_col,
    )

    original_texts = df["text"].astype(str).tolist()
    if title_col and title_col in df.columns:
        combined_texts = [
            f"{title} — {text}" for title, text in zip(df[title_col].astype(str), original_texts)
        ]
    else:
        combined_texts = original_texts
    cleaned_texts = [basic_clean(text) for text in combined_texts]

    if isinstance(min_df, int) and min_df > len(cleaned_texts):
        console.print(
            f"[yellow]min_df={min_df} is larger than document count; using min_df=1.[/yellow]"
        )
        min_df = 1

    console.print("[bold]Vectorizing with TF-IDF...[/bold]")
    vectorizer, matrix = _vectorize(
        cleaned_texts,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
    )

    console.print(f"[bold]Clustering into {k} groups...[/bold]")
    kmeans = _cluster(matrix, k=k)
    labels = kmeans.labels_.tolist()

    silhouette = _compute_silhouette(matrix, labels, k=k)
    top_terms = _top_terms(vectorizer, kmeans, top_n=top_n)
    cluster_sizes = {int(cid): int((kmeans.labels_ == cid).sum()) for cid in range(k)}
    representative_doc_ids = _representative_docs(kmeans, matrix, df["doc_id"].tolist())

    df_out = df.copy()
    df_out["cluster_id"] = labels
    df_out["snippet"] = [
        _make_snippet(text) for text in original_texts
    ]

    output_id_col = None
    if "item_id" in df_out.columns:
        output_id_col = "item_id"
    elif id_col in df_out.columns and id_col != "doc_id":
        output_id_col = id_col
    cols = [output_id_col or "doc_id"]
    if title_col and title_col in df_out.columns:
        cols.append(title_col)
    cols.extend(["cluster_id", "snippet"])
    if "topic_true" in df_out.columns:
        cols.append("topic_true")
    clustered_docs = df_out[cols]

    metrics = {
        "k": k,
        "n_docs": int(len(df_out)),
        "silhouette_score": silhouette,
    }

    cluster_purity = None
    if "topic_true" in df_out.columns:
        cluster_purity = {}
        for cluster_id, group in df_out.groupby("cluster_id"):
            counts = group["topic_true"].value_counts()
            if counts.empty:
                continue
            top_topic = counts.index[0]
            percent = float(counts.iloc[0] / counts.sum() * 100)
            cluster_purity[int(cluster_id)] = {
                "top_topic": str(top_topic),
                "percent": round(percent, 2),
            }

    results = ClusterResults(
        clustered_docs=clustered_docs,
        metrics=metrics,
        cluster_sizes=cluster_sizes,
        top_terms=top_terms,
        representative_doc_ids=representative_doc_ids,
        cluster_purity=cluster_purity,
    )

    write_outputs(results, out_dir)
    return results


def write_outputs(results: ClusterResults, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    clustered_path = out_dir / "clustered_docs.csv"
    clusters_path = out_dir / "clusters.json"
    report_path = out_dir / "report.md"

    results.clustered_docs.to_csv(clustered_path, index=False)

    clusters_payload = {
        "metrics": results.metrics,
        "cluster_sizes": results.cluster_sizes,
        "top_terms": results.top_terms,
        "representative_doc_ids": results.representative_doc_ids,
    }
    if results.cluster_purity is not None:
        clusters_payload["cluster_purity"] = results.cluster_purity
    clusters_path.write_text(json.dumps(clusters_payload, indent=2), encoding="utf-8")

    report_lines = [
        "# Clustering Report",
        "",
        f"- Documents: {results.metrics['n_docs']}",
        f"- Clusters (k): {results.metrics['k']}",
        f"- Silhouette score: {results.metrics['silhouette_score']}",
        "",
        "## Clusters",
    ]
    for cluster_id, size in results.cluster_sizes.items():
        top = ", ".join(results.top_terms.get(cluster_id, []))
        rep = results.representative_doc_ids.get(cluster_id, "n/a")
        report_lines.extend(
            [
                f"### Cluster {cluster_id}",
                f"- Size: {size}",
                f"- Top terms: {top}",
                f"- Representative doc: {rep}",
                "",
            ]
        )

    if results.cluster_purity:
        report_lines.extend(
            [
                "## Cluster Purity (topic_true)",
                "",
                "| Cluster | Top topic | Percent |",
                "| --- | --- | --- |",
            ]
        )
        for cluster_id, info in results.cluster_purity.items():
            report_lines.append(
                f"| {cluster_id} | {info['top_topic']} | {info['percent']}% |"
            )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    console.print(f"[green]Wrote outputs to[/] {out_dir}")
