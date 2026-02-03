"""Typer CLI entrypoint for nlp-cluster-extract."""

from __future__ import annotations

from pathlib import Path
import random

import pandas as pd
import typer
from rich import print

from nlp_cluster_extract.clustering import run_clustering

app = typer.Typer(add_completion=False, no_args_is_help=True)

TOPICS = {
    "politics/government": {
        "headline": [
            "Budget committee debates new spending rules",
            "City council passes zoning reforms",
            "Senate hearing focuses on data privacy",
            "Governor signs regional transit pact",
            "Mayoral race heats up over housing plan",
        ],
        "keywords": ["policy", "committee", "public", "agency", "regulation", "election", "bill"],
    },
    "business/markets": {
        "headline": [
            "Retail chain reports surprise earnings",
            "Manufacturers warn of supply shortages",
            "Investors rally after merger talks",
            "Bank launches new small-business loan",
            "Analysts track volatility in commodities",
        ],
        "keywords": ["shares", "revenue", "quarter", "market", "investment", "inflation", "costs"],
    },
    "technology/ai": {
        "headline": [
            "AI startup releases workflow assistant",
            "Chipmaker unveils energy-efficient server",
            "Researchers test safety rules for AI tools",
            "Cloud provider expands data center capacity",
            "New app uses vision AI for quality checks",
        ],
        "keywords": ["model", "algorithm", "compute", "software", "automation", "cloud", "data"],
    },
    "sports": {
        "headline": [
            "Rival teams trade late goals in opener",
            "Coach reshuffles lineup after injury",
            "League announces playoff schedule",
            "Rookie scores in debut tournament",
            "Fans pack stadium for derby match",
        ],
        "keywords": ["match", "score", "season", "training", "league", "tournament", "fans"],
    },
    "health/medicine": {
        "headline": [
            "Hospital pilots new telehealth program",
            "Researchers publish vaccine trial update",
            "Clinic expands mental health services",
            "Doctors revise treatment guidelines",
            "Health insurers adjust coverage rules",
        ],
        "keywords": ["patient", "care", "trial", "diagnosis", "treatment", "coverage", "costs"],
    },
    "climate/energy": {
        "headline": [
            "Utility invests in grid storage upgrades",
            "Coastal city announces flood resilience plan",
            "Solar project clears environmental review",
            "Heat wave drives demand for cooling",
            "Energy regulators review emissions targets",
        ],
        "keywords": ["emissions", "renewables", "grid", "climate", "carbon", "resilience", "energy"],
    },
    "crime/legal": {
        "headline": [
            "Court hears appeal in fraud case",
            "Police unit expands cybercrime task force",
            "State files lawsuit over unsafe products",
            "Judge sets trial date in bribery probe",
            "Attorneys debate new sentencing rules",
        ],
        "keywords": ["court", "trial", "investigation", "evidence", "charges", "legal", "appeal"],
    },
    "travel/lifestyle": {
        "headline": [
            "Airline adds nonstop routes for summer",
            "Tourism board promotes off-season visits",
            "City opens new waterfront promenade",
            "Travelers seek flexible booking policies",
            "Hotel group refreshes loyalty perks",
        ],
        "keywords": ["trip", "itinerary", "hotel", "flight", "tourism", "booking", "weekend"],
    },
}

SOURCES = [
    "Metro Chronicle",
    "Harbor Daily",
    "Summit Report",
    "Capital Ledger",
    "Lakeside Times",
    "Ridgeview Journal",
    "Civic Pulse",
    "Market Wire",
]

OVERLAP_PHRASES = [
    "lawmakers consider AI regulation",
    "health costs pressure employers",
    "climate policy shapes energy markets",
    "travel demand rises after sports tournament",
    "cybersecurity rules impact hospitals",
]


def _recent_date(days_back: int = 120) -> str:
    from datetime import date, timedelta

    offset = random.randint(0, days_back)
    return (date.today() - timedelta(days=offset)).isoformat()


def _make_blurb(topic: str) -> str:
    keywords = random.sample(TOPICS[topic]["keywords"], k=3)
    overlap = random.choice(OVERLAP_PHRASES) if random.random() < 0.15 else ""
    core = (
        f"Officials said the {keywords[0]} plan targets {keywords[1]} goals while "
        f"monitoring {keywords[2]} impacts."
    )
    if overlap:
        return f"{core} Analysts noted that {overlap}."
    return f"{core} The update drew reactions from local stakeholders."


def _generate_docs(count: int = 160) -> pd.DataFrame:
    topics = list(TOPICS.keys())
    rows: list[dict[str, str]] = []
    per_topic = max(1, count // len(topics))
    item_id = 1

    for topic in topics:
        for _ in range(per_topic):
            headline = random.choice(TOPICS[topic]["headline"])
            rows.append(
                {
                    "item_id": f"item_{item_id:03d}",
                    "topic_true": topic,
                    "headline": headline,
                    "blurb": _make_blurb(topic),
                    "source": random.choice(SOURCES),
                    "published_date": _recent_date(),
                }
            )
            item_id += 1

    while len(rows) < count:
        topic = random.choice(topics)
        headline = random.choice(TOPICS[topic]["headline"])
        rows.append(
            {
                "item_id": f"item_{item_id:03d}",
                "topic_true": topic,
                "headline": headline,
                "blurb": _make_blurb(topic),
                "source": random.choice(SOURCES),
                "published_date": _recent_date(),
            }
        )
        item_id += 1

    return pd.DataFrame(rows)


@app.command("sample-data")
def sample_data(
    output: Path = typer.Option(
        Path("data/sample_news_blurbs.csv"),
        "--output",
        "-o",
        help="Output CSV path.",
    ),
    count: int = typer.Option(160, "--count", help="Number of documents to generate."),
) -> None:
    """Generate a synthetic sample dataset."""
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        print(f"[yellow]Sample data already exists:[/] {output}")
        return

    df = _generate_docs(count=count)
    df.to_csv(output, index=False)
    print(f"[green]Wrote[/] {len(df)} documents to {output}")


@app.command("cluster")
def cluster(
    input_path: Path | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Input CSV file path.",
    ),
    input_folder: Path | None = typer.Option(
        None,
        "--input-folder",
        help="Input folder containing .txt files.",
    ),
    text_col: str = typer.Option("text", "--text-col", help="Text column in CSV."),
    title_col: str | None = typer.Option("title", "--title-col", help="Title column in CSV."),
    id_col: str = typer.Option("doc_id", "--id-col", help="Document id column in CSV."),
    k: int = typer.Option(6, "--k", help="Number of clusters."),
    min_df: float = typer.Option(2, "--min-df", help="Minimum document frequency."),
    max_df: float = typer.Option(0.9, "--max-df", help="Maximum document frequency."),
    ngram_range: str = typer.Option("1,2", "--ngram-range", help="N-gram range, e.g. 1,2."),
    top_n: int = typer.Option(8, "--top-n", help="Top terms per cluster."),
    out_dir: Path = typer.Option(Path("outputs/run1"), "--out", "-o", help="Output folder."),
) -> None:
    """Run TF-IDF + KMeans clustering and write outputs."""
    if input_path and input_folder:
        raise typer.BadParameter("Use either --input or --input-folder, not both.")
    if not input_path and not input_folder:
        raise typer.BadParameter("Provide --input CSV path or --input-folder.")

    try:
        parts = [int(p.strip()) for p in ngram_range.replace("-", ",").split(",") if p.strip()]
        if len(parts) == 1:
            parts = [parts[0], parts[0]]
        if len(parts) != 2:
            raise ValueError
        ngram_tuple = (parts[0], parts[1])
    except ValueError as exc:
        raise typer.BadParameter("Use --ngram-range like 1,2") from exc

    # scikit-learn requires int for min_df/max_df when >= 1
    if isinstance(min_df, float) and min_df.is_integer() and min_df >= 1:
        min_df = int(min_df)
    if isinstance(max_df, float) and max_df.is_integer() and max_df >= 1:
        max_df = int(max_df)

    run_clustering(
        input_path=input_path,
        input_folder=input_folder,
        out_dir=out_dir,
        text_col=text_col,
        title_col=title_col,
        id_col=id_col,
        k=k,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_tuple,
        top_n=top_n,
    )


@app.command("extract")
def extract(
    input_path: Path | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Input CSV file path.",
    ),
    input_folder: Path | None = typer.Option(
        None,
        "--input-folder",
        help="Input folder containing .txt files.",
    ),
    text_col: str = typer.Option("text", "--text-col", help="Text column in CSV."),
    title_col: str | None = typer.Option("title", "--title-col", help="Title column in CSV."),
    id_col: str = typer.Option("doc_id", "--id-col", help="Document id column in CSV."),
    top_k: int = typer.Option(12, "--top-k", help="Top keywords per document."),
    use_ner: bool = typer.Option(False, "--use-ner", help="Enable spaCy NER if available."),
    out_dir: Path = typer.Option(Path("outputs/extract1"), "--out", "-o", help="Output folder."),
) -> None:
    """Run keyword + pattern extraction and write outputs."""
    if input_path and input_folder:
        raise typer.BadParameter("Use either --input or --input-folder, not both.")
    if not input_path and not input_folder:
        raise typer.BadParameter("Provide --input CSV path or --input-folder.")

    from nlp_cluster_extract.extraction import run_extraction

    run_extraction(
        input_path=input_path,
        input_folder=input_folder,
        out_dir=out_dir,
        text_col=text_col,
        title_col=title_col,
        id_col=id_col,
        top_k=top_k,
        use_ner=use_ner,
    )


if __name__ == "__main__":
    app()
