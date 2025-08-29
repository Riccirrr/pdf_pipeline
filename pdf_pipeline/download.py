"""Asynchronous PDF downloader with enhanced link resolution and validation.

This module implements high‑concurrency downloading of PDF files from
arbitrary URLs with advanced features including:
- URL normalization and HTTPS enforcement
- PDF link extraction from HTML pages
- Filename sanitization and safe handling
- Content validation and minimum size checks
- Automatic referer handling for common sites

The primary entry points are :func:`download_pdfs` and helper functions
for URL resolution and validation.

Example usage::

    import asyncio
    from pdf_pipeline.download import download_pdfs, DownloadConfig

    urls = [
        "https://example.com/doc1.pdf",
        "https://cninfo.com.cn/detail/page?stockCode=000001&announcementId=123",
    ]
    config = DownloadConfig(timeout=30, retries=2)
    results = asyncio.run(download_pdfs(urls, "/tmp/pdfs", config=config))
    for url, success, path_or_error in results:
        if success:
            print(f"Downloaded {url} to {path_or_error}")
        else:
            print(f"Failed to download {url}: {path_or_error}")
"""

from __future__ import annotations

import asyncio
import re
import html
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, parse_qs
from dataclasses import dataclass

import aiohttp
from aiohttp import ClientError, ClientTimeout
from yarl import URL

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/110.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

@dataclass
class DownloadConfig:
    """Configuration for PDF download operations."""
    timeout: int = 30
    retries: int = 2
    backoff_base: float = 1.5
    backoff_jitter: float = 0.1
    max_concurrency: int = 5
    per_host_limit: int = 6

# Filename sanitization
FORBIDDEN = '<>:"/\\|?*'

def _sanitize_filename(name: str, maxlen: int = 200) -> str:
    """Sanitize filename by replacing forbidden characters."""
    name = "".join(('_' if c in FORBIDDEN else c) for c in name)
    return name[:maxlen]

def _name_from_cninfo_query(url: str) -> str | None:
    """Generate filename from cninfo.com.cn query parameters."""
    qs = parse_qs(urlsplit(url).query)
    code = (qs.get("stockCode") or [""])[0]
    ann  = (qs.get("announcementId") or [""])[0]
    t    = (qs.get("announcementTime") or [""])[0].split(" ")[0].replace("-", "")
    parts = [p for p in (code, ann, t) if p]
    return _sanitize_filename("_".join(parts) + ".pdf") if parts else None

def _build_cninfo_static_url_from_detail(url: str) -> str | None:
    """Build direct static URL from cninfo detail page parameters."""
    # detail?stockCode=...&announcementId=...&announcementTime=YYYY-MM-DD ...
    qs = parse_qs(urlsplit(url).query)
    ann = (qs.get("announcementId") or [""])[0]
    tm  = (qs.get("announcementTime") or [""])[0].split(" ")[0]
    if ann and tm:
        # Note: cninfo direct links usually use uppercase .PDF
        return f"https://static.cninfo.com.cn/finalpage/{tm}/{ann}.PDF"
    return None

def _is_pdf_response(resp: aiohttp.ClientResponse) -> bool:
    """Check if response contains PDF content based on headers."""
    ct = (resp.headers.get("Content-Type") or "").lower()
    cd = (resp.headers.get("Content-Disposition") or "").lower()
    return ("pdf" in ct) or (".pdf" in cd)

def _default_referer_for(host: str) -> str | None:
    """Get default referer for specific hosts."""
    if host.endswith("cninfo.com.cn"):
        return "https://www.cninfo.com.cn/"
    return None

async def _read_text_safely(resp: aiohttp.ClientResponse, limit_bytes: int = 2_000_000) -> str:
    """Read response text with size limit to avoid memory issues."""
    chunks = []
    total = 0
    async for ch in resp.content.iter_chunked(64 * 1024):
        total += len(ch)
        chunks.append(ch)
        if total >= limit_bytes:
            break
    return b"".join(chunks).decode(errors="ignore")

def _extract_pdf_candidates(base_url: URL, html_text: str) -> list[str]:
    """Extract potential PDF URLs from HTML content."""
    patterns = [
        r'href=["\']([^"\']+\.pdf)["\']',
        r"window\.open\(['\"]([^'\"]+\.pdf)['\"]\)",
        r'href=["\']([^"\']*download[^"\']*announcementId=\d+[^"\']*)["\']',
        r'src=["\']([^"\']+\.pdf)["\']',
        r'href=["\'](/finalpage/[^"\']+\.PDF)["\']',
    ]
    cands: set[str] = set()
    for pat in patterns:
        for m in re.finditer(pat, html_text, flags=re.IGNORECASE):
            href = html.unescape(m.group(1))
            try:
                absu = str(URL(href, encoded=False).join(base_url))
            except Exception:
                # Fallback to manual URL joining
                absu = str(base_url.with_path(href)) if href.startswith("/") else str(base_url / href)
            cands.add(absu)
    return list(cands)

async def _resolve_cninfo_pdf_url(session: aiohttp.ClientSession, detail_url: str, headers: dict) -> str | None:
    """Resolve PDF URL from cninfo detail page using multiple strategies."""
    # 1) Direct static URL construction (using announcementId + announcementTime)
    static_url = _build_cninfo_static_url_from_detail(detail_url)
    if static_url:
        try:
            async with session.head(static_url, headers=headers, allow_redirects=True) as r:
                if r.status < 400 and _is_pdf_response(r):
                    return str(r.url)
        except Exception:
            pass

    # 2) Parse detail page HTML (fallback)
    u = URL(detail_url, encoded=False)
    h = dict(headers)
    ref = h.get("Referer") or _default_referer_for(u.host)
    if ref: 
        h["Referer"] = ref
    async with session.get(str(u), headers=h, allow_redirects=True) as resp:
        if _is_pdf_response(resp):
            return str(resp.url)
        text = await _read_text_safely(resp)
        base = URL(str(resp.url))
        for cand in _extract_pdf_candidates(base, text):
            try:
                async with session.head(cand, headers=h, allow_redirects=True) as r2:
                    if r2.status < 400 and _is_pdf_response(r2):
                        return str(r2.url)
            except Exception:
                continue

    # 3) Official query API /new/hisAnnouncement/query, match by announcementId for adjunctUrl
    qs = parse_qs(urlsplit(detail_url).query)
    stock_code = (qs.get("stockCode") or [""])[0]
    org_id     = (qs.get("orgId") or [""])[0]
    ann_id     = (qs.get("announcementId") or [""])[0]
    if not (stock_code and org_id and ann_id):
        return None

    # column/plate inference: 6xxxx belongs to Shanghai Stock Exchange (sse), others default to Shenzhen (szse)
    is_sse = stock_code.startswith("6")
    column = "sse" if is_sse else "szse"
    plate  = "sh" if is_sse else "sz"

    form = {
        "pageNum": "1", "pageSize": "30",
        "tabName": "fulltext",
        "column": column,
        "plate": plate,
        "stock": f"{stock_code},{org_id}",
        "seDate": "", "searchkey": "", "secid": "",
        "sortName": "", "sortType": "", "isHLtitle": "true",
    }
    api = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
    api_headers = dict(h)
    api_headers.update({
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
    })
    try:
        async with session.post(api, data=form, headers=api_headers) as r:
            if r.status >= 400: 
                return None
            data = await r.json(content_type=None)
            anns = data.get("announcements") or []
            for it in anns:
                if str(it.get("announcementId")) == str(ann_id):
                    adj = it.get("adjunctUrl")
                    if adj:
                        return f"https://static.cninfo.com.cn/{adj}"
    except Exception:
        pass
    return None

async def _find_pdf_url(session: aiohttp.ClientSession, url: str, headers: dict) -> str | None:
    """Find actual PDF URL by following redirects and parsing HTML if needed."""
    # Normalize to HTTPS and handle encoding
    u = URL(url, encoded=False)
    if u.scheme not in ("http", "https"):
        u = u.with_scheme("https")
    if u.scheme == "http":
        u = u.with_scheme("https")

    referer = headers.get("Referer") or _default_referer_for(u.host)
    h = dict(headers)
    if referer:
        h["Referer"] = referer

    async with session.get(str(u), headers=h, allow_redirects=True) as resp:
        if _is_pdf_response(resp):
            return str(resp.url)

        # Not a PDF, try to find PDF links in HTML
        text = await _read_text_safely(resp)
        base = URL(str(resp.url))
        for cand in _extract_pdf_candidates(base, text):
            try:
                async with session.head(cand, headers=h, allow_redirects=True) as r2:
                    if _is_pdf_response(r2):
                        return str(r2.url)
            except Exception:
                continue

        # Fallback: try GET on first candidate
        for cand in _extract_pdf_candidates(base, text):
            try:
                async with session.get(cand, headers=h, allow_redirects=True) as r2:
                    if _is_pdf_response(r2):
                        return str(r2.url)
            except Exception:
                continue
    return None

def _suggest_filename_from_headers_or_query(url: str, headers: aiohttp.typedefs.LooseHeaders) -> str:
    """Generate filename from response headers or URL parameters."""
    # Try Content-Disposition first
    cd = headers.get("Content-Disposition") if headers else None
    if cd:
        m = re.search(r'filename\*=UTF-8\'\'([^;]+)|filename="?([^\";]+)"?', cd, re.IGNORECASE)
        if m:
            filename = m.group(1) or m.group(2)
            return _sanitize_filename(Path(filename).name)

    # Try cninfo query parameters
    name = _name_from_cninfo_query(url)
    if name:
        return name

    # Fallback: use URL path
    tail = urlsplit(url).path.rstrip("/").split("/")[-1] or "download.pdf"
    if not tail.lower().endswith(".pdf"):
        tail += ".pdf"
    return _sanitize_filename(tail)

async def _sleep_backoff(attempt: int, base: float, jitter: float):
    """Sleep with exponential backoff and jitter."""
    import random
    delay = base ** attempt
    jitter_amount = delay * jitter * random.random()
    await asyncio.sleep(delay + jitter_amount)

async def _download_pdf(
    session: aiohttp.ClientSession,
    pdf_url: str,
    dest_dir: Path,
    cfg: DownloadConfig,
    headers: Optional[Dict[str, str]],
    min_bytes: int = 1024,  # Minimum file size to consider valid
) -> Tuple[bool, str]:
    """Download PDF file with validation."""
    merged_headers = dict(DEFAULT_HEADERS)
    if headers:
        merged_headers.update(headers)
    referer = merged_headers.get("Referer") or _default_referer_for(URL(pdf_url).host)
    if referer:
        merged_headers["Referer"] = referer

    async with session.get(pdf_url, headers=merged_headers, timeout=cfg.timeout, allow_redirects=True) as resp:
        if resp.status >= 400 or not _is_pdf_response(resp):
            return False, f"Not a PDF or HTTP {resp.status}"
        
        filename = _suggest_filename_from_headers_or_query(pdf_url, resp.headers)
        path = dest_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        
        size = 0
        with open(path, "wb") as f:
            async for chunk in resp.content.iter_chunked(64 * 1024):
                f.write(chunk)
                size += len(chunk)
        
        if size < min_bytes:
            try: 
                path.unlink()
            except Exception: 
                pass
            return False, f"Downloaded size {size} < {min_bytes}"
        
        return True, str(path)

async def _download_one(
    session: aiohttp.ClientSession,
    url: str,
    dest_dir: Path,
    cfg: DownloadConfig,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[str, bool, str]:
    """Download single URL with retries and error handling."""
    merged_headers = dict(DEFAULT_HEADERS)
    if headers:
        merged_headers.update(headers)

    for attempt in range(cfg.retries + 1):
        try:
            u = URL(url, encoded=False)
            pdf_url = None
            
            # Use enhanced cninfo resolution for detail pages
            if u.host and u.host.endswith("cninfo.com.cn") and "/new/disclosure/detail" in u.path:
                pdf_url = await _resolve_cninfo_pdf_url(session, url, merged_headers)
            
            # Fallback to generic resolution if cninfo-specific failed or not applicable
            if not pdf_url:
                pdf_url = await _find_pdf_url(session, url, merged_headers)
            
            # Final fallback: treat as direct PDF URL
            if not pdf_url:
                pdf_url = str(u)
            
            ok, msg = await _download_pdf(session, pdf_url, dest_dir, cfg, merged_headers)
            if ok:
                return (url, True, msg)
            raise RuntimeError(msg)
            
        except Exception as e:
            if attempt < cfg.retries:
                await _sleep_backoff(attempt + 1, cfg.backoff_base, cfg.backoff_jitter)
                continue
            return (url, False, f"{type(e).__name__}: {e}")

async def download_pdfs(
    urls: Sequence[str],
    dest_dir: str | Path,
    config: Optional[DownloadConfig] = None,
    headers: Optional[Dict[str, str]] = None,
    max_download_concurrency: Optional[int] = None,
    per_host_limit: Optional[int] = None,
    timeout: Optional[int] = None,
    retries: Optional[int] = None,
    backoff_base: Optional[float] = None,
    backoff_jitter: Optional[float] = None,
) -> List[Tuple[str, bool, str]]:
    """Download multiple PDFs with advanced resolution and validation.
    
    Parameters
    ----------
    urls : Sequence[str]
        URLs to download (can be direct PDF links or detail pages)
    dest_dir : str | Path
        Destination directory for downloads
    config : DownloadConfig, optional
        Download configuration object
    headers : dict, optional
        Additional HTTP headers
    max_download_concurrency : int, optional
        Override config max concurrency
    per_host_limit : int, optional  
        Override config per-host limit
    timeout : int, optional
        Override config timeout
    retries : int, optional
        Override config retries
    backoff_base : float, optional
        Override config backoff base
    backoff_jitter : float, optional
        Override config backoff jitter
        
    Returns
    -------
    List[Tuple[str, bool, str]]
        List of (url, success, path_or_error) tuples
    """
    if config is None:
        config = DownloadConfig()
    
    # Override config with explicit parameters
    if max_download_concurrency is not None:
        config.max_concurrency = max_download_concurrency
    if per_host_limit is not None:
        config.per_host_limit = per_host_limit
    if timeout is not None:
        config.timeout = timeout
    if retries is not None:
        config.retries = retries
    if backoff_base is not None:
        config.backoff_base = backoff_base
    if backoff_jitter is not None:
        config.backoff_jitter = backoff_jitter
    
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    merged_headers = dict(DEFAULT_HEADERS)
    if headers:
        merged_headers.update(headers)
    
    connector = aiohttp.TCPConnector(
        limit=config.max_concurrency,
        limit_per_host=config.per_host_limit,
        enable_cleanup_closed=True
    )
    timeout_ctx = ClientTimeout(total=config.timeout)
    
    async with aiohttp.ClientSession(
        headers=merged_headers,
        connector=connector, 
        timeout=timeout_ctx
    ) as session:
        semaphore = asyncio.Semaphore(config.max_concurrency)
        
        async def download_with_semaphore(url: str) -> Tuple[str, bool, str]:
            async with semaphore:
                return await _download_one(session, url, dest_path, config, headers)
        
        tasks = [download_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks)
    
    return list(results)

# Legacy compatibility
async def download_files(
    urls: Sequence[str],
    dest_dir: str,
    max_concurrency: int = 5,
    headers: Optional[dict] = None,
    timeout: Optional[int] = 30,
    retries: int = 2,
) -> List[Tuple[str, bool, str]]:
    """Legacy compatibility function."""
    config = DownloadConfig(
        max_concurrency=max_concurrency,
        timeout=timeout or 30,
        retries=retries
    )
    return await download_pdfs(urls, dest_dir, config=config, headers=headers)

__all__ = ["download_pdfs", "download_files", "DownloadConfig"]