"""End‑to‑end pipeline orchestrating downloading, parsing and organising PDFs.

This module exposes a high‑level `run_pipeline` function which accepts a
list of PDF URLs, downloads them concurrently, parses their contents and
saves structured JSON outputs alongside any extracted images.  It uses
the :mod:`pdf_pipeline.download`, :mod:`pdf_pipeline.parser` and
:mod:`pdf_pipeline.organizer` submodules under the hood.

Running this module as a script from the command line offers a simple
interface for processing a newline‑separated list of URLs::

    python -m pdf_pipeline.pipeline --urls urls.txt --download-dir downloads --output-dir parsed

The script will read URLs from ``urls.txt``, download each PDF
concurrently into ``downloads/``, parse them concurrently and write
JSON results into ``parsed/json`` while images live in ``parsed/images``.
The summary of successes and failures is printed at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from .download import download_pdfs, DownloadConfig
from .parser import parse_pdfs
from .organizer import save_to_json, clean_tables_in_result


async def run_pipeline(
    urls: Sequence[str],
    download_dir: str,
    output_dir: str,
    download_concurrency: int = 5,
    parse_concurrency: int = 4,
    retries: int = 2,
    enable_ocr: bool = False,
    ocr_all_pages: bool = False,
    clean_tables: bool = True,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    per_host_limit: int = 6,
    backoff_base: float = 1.5,
    backoff_jitter: float = 0.1,
) -> Tuple[int, int]:
    """Run the full pipeline on a list of PDF URLs.

    Parameters
    ----------
    urls : sequence of str
        List of PDF URLs to download and process.
    download_dir : str
        Directory to store downloaded PDF files.
    output_dir : str
        Base directory for parsed outputs; images and JSON will be
        placed underneath this folder.
    download_concurrency : int, optional
        Maximum number of concurrent downloads.
    parse_concurrency : int, optional
        Maximum number of concurrent parsing tasks.
    retries : int, optional
        Number of retry attempts for each download.
    clean_tables : bool, optional
        Whether to apply table cleaning to parsed results before saving.
        Default is True.

    Returns
    -------
    tuple of (int, int)
        A tuple containing the number of successfully processed files
        and the number of failures (download or parse).
    """
    # Step 1: Download all PDFs
    download_results = await download_pdfs(
        urls,
        dest_dir=download_dir,
        max_download_concurrency=download_concurrency,
        retries=retries,
        headers=headers,
        timeout=timeout,
        per_host_limit=per_host_limit,
        backoff_base=backoff_base,
        backoff_jitter=backoff_jitter,
    )
    # Separate successes and failures
    downloaded_files: List[str] = []
    failed_downloads: List[str] = []
    for url, success, message in download_results:
        if success:
            downloaded_files.append(message)
        else:
            failed_downloads.append(f"{url}: {message}")
    # Step 2: Parse downloaded PDFs concurrently
    parse_results = await parse_pdfs(
        downloaded_files,
        output_dir=output_dir,
        max_concurrency=parse_concurrency,
        enable_ocr=enable_ocr,
        ocr_all_pages=ocr_all_pages,
    )
    # Collect successful parse data and track errors
    parsed_data = []
    failed_parses = []
    for file_name, success, data in parse_results:
        if success:
            parsed_data.append(data)
        else:
            failed_parses.append(f"{file_name}: {data.get('error')}")
    # Step 3: Clean tables if requested and save JSON outputs
    if clean_tables:
        parsed_data = [clean_tables_in_result(item) for item in parsed_data]
    json_dir = str(Path(output_dir) / "json")
    save_to_json(parsed_data, json_dir)
    # Print or return summary
    total_success = len(parsed_data)
    total_failures = len(failed_downloads) + len(failed_parses)
    if failed_downloads or failed_parses:
        sys.stderr.write("Some files failed to download or parse:\n")
        for err in failed_downloads + failed_parses:
            sys.stderr.write(f"  {err}\n")
    return total_success, total_failures


def _read_urls_from_file(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _auto_referer_for_urls(urls: Sequence[str]) -> Optional[str]:
    """Auto-detect referer for URLs based on domain patterns."""
    for url in urls:
        host = urlparse(url).netloc
        if host.endswith("cninfo.com.cn"):
            return "https://www.cninfo.com.cn/"
    return None


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and parse PDF announcements at scale.")
    parser.add_argument(
        "--urls",
        required=True,
        help="Path to a text file containing one PDF URL per line.",
    )
    parser.add_argument(
        "--download-dir",
        required=True,
        help="Directory to store downloaded PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store parsed outputs (JSON and images).",
    )
    parser.add_argument(
        "--download-concurrency",
        type=int,
        default=5,
        help="Maximum number of simultaneous downloads.",
    )
    parser.add_argument(
        "--parse-concurrency",
        type=int,
        default=4,
        help="Maximum number of simultaneous parsing tasks.",
    )
    parser.add_argument(
        "--per-host-limit",
        type=int,
        default=6,
        help="Maximum concurrent connections per host.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retry attempts per URL.",
    )
    parser.add_argument(
        "--backoff-base",
        type=float,
        default=1.5,
        help="Exponential backoff base for retries.",
    )
    parser.add_argument(
        "--backoff-jitter",
        type=float,
        default=0.1,
        help="Jitter factor for backoff delays.",
    )
    parser.add_argument(
        "--user-agent",
        help="Custom User-Agent header.",
    )
    parser.add_argument(
        "--referer",
        help="Custom Referer header.",
    )
    parser.add_argument("--enable-ocr", action="store_true", help="启用 OCR 回退（PP-Structure）")
    parser.add_argument("--ocr-all-pages", action="store_true", help="对所有页面都做 OCR（默认仅在扫描/失败时回退）")
    parser.add_argument("--no-clean-tables", action="store_true", help="禁用表格清洗功能（默认启用）")
    args = parser.parse_args(argv)
    urls = _read_urls_from_file(args.urls)
    
    # Build custom headers
    custom_headers = {}
    if args.user_agent:
        custom_headers["User-Agent"] = args.user_agent
    
    # Auto-detect referer or use custom one
    if args.referer:
        custom_headers["Referer"] = args.referer
    else:
        auto_ref = _auto_referer_for_urls(urls)
        if auto_ref and "Referer" not in custom_headers:
            custom_headers["Referer"] = auto_ref
    
    total_success, total_failures = asyncio.run(
        run_pipeline(
            urls,
            download_dir=args.download_dir,
            output_dir=args.output_dir,
            download_concurrency=args.download_concurrency,
            parse_concurrency=args.parse_concurrency,
            enable_ocr=args.enable_ocr,
            ocr_all_pages=args.ocr_all_pages,
            clean_tables=not args.no_clean_tables,
            headers=custom_headers or None,
            timeout=args.timeout,
            per_host_limit=args.per_host_limit,
            backoff_base=args.backoff_base,
            backoff_jitter=args.backoff_jitter,
        )
    )
    print(f"Successfully processed {total_success} file(s).")
    if total_failures:
        print(f"{total_failures} file(s) failed to process.")


if __name__ == "__main__":
    main()