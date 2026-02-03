import json
from pathlib import Path

import pandas as pd

from nlp_cluster_extract.clustering import run_clustering
from nlp_cluster_extract.preprocess import basic_clean
from nlp_cluster_extract.cli import _generate_docs


def test_basic_clean_smoke():
    text = "Hello, world!!!"
    assert basic_clean(text) == "hello world"


def _write_sample_csv(path: Path, rows: int = 60) -> None:
    df = _generate_docs(count=rows)
    df.to_csv(path, index=False)


def test_clustering_runs_on_sample_data(tmp_path: Path):
    input_csv = tmp_path / "sample_docs.csv"
    _write_sample_csv(input_csv, rows=60)

    out_dir = tmp_path / "outputs"
    results = run_clustering(
        input_path=input_csv,
        input_folder=None,
        out_dir=out_dir,
        text_col="blurb",
        title_col="headline",
        id_col="item_id",
        k=6,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        top_n=5,
    )

    assert results.clustered_docs.shape[0] == 60
    assert (out_dir / "clustered_docs.csv").exists()


def test_clustering_outputs_have_expected_keys(tmp_path: Path):
    input_csv = tmp_path / "sample_docs.csv"
    _write_sample_csv(input_csv, rows=30)

    out_dir = tmp_path / "outputs"
    run_clustering(
        input_path=input_csv,
        input_folder=None,
        out_dir=out_dir,
        text_col="blurb",
        title_col="headline",
        id_col="item_id",
        k=3,
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 1),
        top_n=3,
    )

    clustered = pd.read_csv(out_dir / "clustered_docs.csv")
    assert {"item_id", "headline", "cluster_id", "snippet", "topic_true"}.issubset(
        clustered.columns
    )

    clusters_data = json.loads((out_dir / "clusters.json").read_text(encoding="utf-8"))
    assert {"metrics", "cluster_sizes", "top_terms", "representative_doc_ids"}.issubset(
        clusters_data.keys()
    )
