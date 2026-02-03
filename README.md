# nlp-cluster-extract

Portfolio-grade NLP toolkit for document clustering and information extraction.

## Quickstart (Windows)

```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
python scripts\generate_sample_data.py
nlpce --help
```

## What’s included

- TF-IDF + KMeans clustering
- Keyword extraction + entity/PII extraction
- Typer-based CLI
- Synthetic sample data generator
- Streamlit UI demo

## CLI

- `nlpce sample-data` generates `data/sample_news_blurbs.csv` if missing
- `nlpce cluster` runs clustering and writes outputs
- `nlpce extract` runs keyword + pattern extraction

## Project layout

```
./
+- app/
+- data/
+- scripts/
+- src/
¦  +- nlp_cluster_extract/
+- tests/
+- requirements.txt
+- requirements-dev.txt
+- pyproject.toml
+- README.md
```

## Clustering demo

```powershell
nlpce cluster --input data/sample_news_blurbs.csv --text-col blurb --title-col headline --k 8 --out outputs/news_run1
```

Outputs written to `outputs/news_run1`:

- `clustered_docs.csv` with `item_id`, `headline`, `cluster_id`, `snippet`, and `topic_true` (if present)
- `clusters.json` with metrics, cluster sizes, top terms, and representative doc ids
- `report.md` human-readable summary (includes cluster purity if `topic_true` exists)

## Extraction demo

```powershell
nlpce extract --input data/sample_news_blurbs.csv --text-col blurb --out outputs/extract1 --top-k 12
```

Outputs written to `outputs/extract1`:

- `extraction.json` with per-doc keywords and extracted patterns
- `keywords_summary.md` with global keyword counts (and per-cluster highlights if `clustered_docs.csv` exists)

## UI demo

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app\streamlit_app.py
```

If the sample data is missing:

```powershell
nlpce sample-data
```