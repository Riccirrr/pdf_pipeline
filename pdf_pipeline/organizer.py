"""Utilities for organising parsed PDF data.

This module provides helper functions to persist parsed PDF outputs
into JSON files and to aggregate results across multiple documents.
Since parsing may be performed concurrently, organisation is kept
separate so that it can be executed after all parsing completes.

Example::

    from pdf_pipeline.organizer import save_to_json

    parsed_results = [
        {"file_name": "foo.pdf", ...},
        {"file_name": "bar.pdf", ...},
    ]
    save_to_json(parsed_results, output_dir="./output/json")

"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Union, Any

import pandas as pd
from .parser import _clean_table_df  # 复用上面的清洗逻辑


def clean_tables_in_result(result: dict) -> dict:
    """Clean tables in a parsed result dictionary.
    
    This function takes a result dictionary that contains tables with raw_json
    data and adds clean_json and metadata by applying the table cleaning logic.
    Useful for batch processing existing JSON files that were parsed before
    the cleaning functionality was added.
    
    Parameters
    ----------
    result : dict
        A parsed PDF result dictionary containing tables with raw_json data.
        
    Returns
    -------
    dict
        The same result dictionary with clean_json and meta added to each table.
    """
    new_tables = []
    for t in result.get("tables", []):
        try:
            df_raw = pd.read_json(StringIO(t.get("raw_json")), orient="split")
            df_clean, meta = _clean_table_df(df_raw)
            t["clean_json"] = df_clean.to_json(orient="split", force_ascii=False)
            t["meta"] = (t.get("meta") or {}) | meta
            t["shape_clean"] = list(df_clean.shape)
        except Exception as e:
            t["clean_error"] = f"{type(e).__name__}: {e}"
        new_tables.append(t)
    result["tables"] = new_tables
    return result


def save_to_json(results: Iterable[Dict[str, Union[str, dict, list]]], output_dir: str) -> None:
    """Persist parsed PDF results into individual JSON files.

    For each entry in ``results`` this function writes a JSON file named
    after the PDF's stem (base filename without extension) into
    ``output_dir``.  Images are assumed to have already been saved
    during parsing.

    Parameters
    ----------
    results : iterable of dict
        Parsed PDF objects returned from :mod:`pdf_pipeline.parser`.
    output_dir : str
        Directory where JSON files will be written.  Created if not existing.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for result in results:
        if not isinstance(result, dict):
            raise TypeError(f"Expected dict entries, got {type(result)}")
        file_name = result.get("file_name", None)
        if not file_name:
            raise ValueError("Result entry missing 'file_name'")
        stem = Path(file_name).stem
        json_path = out_path / f"{stem}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def aggregate_results(results: Iterable[Dict[str, Union[str, dict, list]]]) -> Dict[str, Any]:
    """Aggregate multiple parsed results into a single dictionary.

    The returned dictionary maps the base filename (without extension)
    to its parsed data.  This may be useful when returning all
    documents in a single JSON structure instead of writing individual
    files.

    Parameters
    ----------
    results : iterable of dict
        Parsed PDF objects.

    Returns
    -------
    dict
        A mapping from base filename to the original parsed dictionary.
    """
    aggregated: Dict[str, Any] = {}
    for result in results:
        file_name = result.get("file_name")
        if not file_name:
            continue
        key = Path(file_name).stem
        aggregated[key] = result
    return aggregated


__all__ = ["save_to_json", "aggregate_results", "clean_tables_in_result"]