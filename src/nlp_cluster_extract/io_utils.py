"""I/O utilities for loading documents."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {"doc_id", "title", "text"}


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load documents from a CSV file with required columns."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {', '.join(sorted(missing))}")

    # Ensure consistent ordering for downstream usage.
    return df[["doc_id", "title", "text"]].copy()


def load_txt_folder(folder: str | Path) -> pd.DataFrame:
    """Load a folder of .txt files into a DataFrame.

    Each file becomes a row with doc_id, title (stem), and text.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    rows: list[dict[str, str]] = []
    for txt_file in sorted(folder_path.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8").strip()
        rows.append(
            {
                "doc_id": txt_file.stem,
                "title": txt_file.stem.replace("_", " ").title(),
                "text": text,
            }
        )

    if not rows:
        raise ValueError(f"No .txt files found in {folder_path}")

    return pd.DataFrame(rows)


def ensure_iterable_texts(texts: Iterable[str]) -> list[str]:
    """Return a list of non-empty strings for downstream processing."""
    cleaned = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
    if not cleaned:
        raise ValueError("No valid text entries provided.")
    return cleaned