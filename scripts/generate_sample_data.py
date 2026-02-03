"""Generate synthetic sample data for the project."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlp_cluster_extract.cli import _generate_docs  # noqa: E402


def main() -> None:
    output = ROOT / "data" / "sample_news_blurbs.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    df = _generate_docs(count=160)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} items to {output}")


if __name__ == "__main__":
    main()
