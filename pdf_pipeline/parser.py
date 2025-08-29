"""PDF document parser.

This module provides utilities to extract structured content from PDF
documents.  It builds on several optional third‑party libraries to
achieve high‑quality results:

* **PyMuPDF4LLM**: Converts PDF pages into Markdown, preserving
  multi‑column layouts, headings and inline formatting【401008996391982†L116-L151】.  If
  unavailable, plain text is extracted using :mod:`fitz` directly.
* **Camelot**: Extracts tabular data from text‑based PDFs.  Tables are
  returned as lists of lists; the library supports both the lattice
  method for well‑defined tables and the stream method for implicit
  tables【633924826990229†L43-L74】.  When Camelot is not installed, the module
  falls back to empty results.
* **PyMuPDF**: Used for image extraction regardless of the presence of
  other libraries.  Each embedded image in the PDF is exported and
  saved into an ``images/`` subdirectory alongside the parsed JSON.
* **PaddleOCR (optional)**: If Camelot fails and PaddleOCR is
  available, developers may integrate the table structure recognition
  pipeline to handle scanned or complex table images【292402067347150†L318-L324】.

The main entry point is :func:`parse_pdf`, which returns a dictionary
containing text blocks, tables and image metadata.  For batch
processing, :func:`parse_pdfs` accepts a list of file paths and uses
thread pools to parallelize parsing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd  # 确保已安装（pandas 在 requirements 里）

try:
    import pymupdf4llm  # type: ignore
    _HAS_PYMUPDF4LLM = True
except ImportError:
    _HAS_PYMUPDF4LLM = False

try:
    import camelot  # type: ignore
    _HAS_CAMELOT = True
except ImportError:
    _HAS_CAMELOT = False

try:
    from paddleocr import PPStructure
    _HAS_PPOCR = True
except ImportError:
    _HAS_PPOCR = False

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise RuntimeError(
        "PyMuPDF (fitz) must be installed to use pdf_pipeline.parser"
    ) from exc

# Optional logging configuration – callers may configure the root logger.
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Attach a basic handler if none exists.  This ensures warnings are visible.
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# === Post-processing helpers ==================================================

_CN_CURRENCY_ALIASES = {
    "人民币": "CNY",
    "RMB": "CNY",
    "元": "CNY",
    "港元": "HKD",
    "美元": "USD",
}

_CN_UNIT_SCALE = {
    "元": Decimal("1"),
    "万元": Decimal("10000"),
    "百万元": Decimal("1000000"),
    "亿元": Decimal("100000000"),
}

_NUM_RE = re.compile(r"^[\s\u00a0\-–—\(\)（）\[\]\+±．\.\,，、0-9%：:]+$")

def _to_halfwidth(text: str) -> str:
    # 全角转半角，清理多种空白
    s = unicodedata.normalize("NFKC", str(text))
    s = s.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    return re.sub(r"[ \t\r\f\v]+", " ", s).strip()

def _looks_like_header_row(values) -> bool:
    # "非数字占比高"的行更像表头
    if not values:
        return False
    nonnum = 0
    for v in values:
        vs = _to_halfwidth(v)
        if not vs:
            nonnum += 1
            continue
        # 含明显中文/字母视为表头片段
        if re.search(r"[A-Za-z\u4e00-\u9fa5]", vs):
            nonnum += 1
        else:
            # 全是符号/数字也可能是单位行，这里不计为非数
            pass
    return nonnum >= max(1, len(values) // 2)

def _detect_unit_and_currency(df: pd.DataFrame):
    """
    扫描前 2~3 行是否含"单位/币种/人民币元/万元"等，并返回：
      {"currency": "CNY", "unit": "元", "scale": Decimal('1'), "row_idx": i or None, "raw": "..."}
    若未检测到返回 None。
    """
    max_scan = min(3, len(df))
    pat = re.compile(r"(币种|单位)[：: ]?\s*([^\s]+)?\s*(人民币|港元|美元|RMB)?\s*(元|万元|百万元|亿元)?", re.I)
    for i in range(max_scan):
        row_text = " ".join([_to_halfwidth(x) for x in df.iloc[i].astype(str).tolist()])
        m = pat.search(row_text)
        if not m:
            # 也支持常见的"单位：人民币元""单位：元"
            if "单位" in row_text or "币种" in row_text:
                # 简单兜底：抓"人民币/港元/美元"和"元/万元/百万元/亿元"
                cur = None
                for k, v in _CN_CURRENCY_ALIASES.items():
                    if k in row_text:
                        cur = v; break
                unit = None
                for u in _CN_UNIT_SCALE.keys():
                    if u in row_text:
                        unit = u; break
                scale = _CN_UNIT_SCALE.get(unit or "元", Decimal("1"))
                return {"currency": cur or "CNY", "unit": unit or "元", "scale": scale,
                        "row_idx": i, "raw": row_text}
            continue
        # 明确命中正则
        grp = m.groups()
        cur_word = grp[2] or grp[1] or ""
        unit_word = grp[3] or "元"
        currency = _CN_CURRENCY_ALIASES.get(cur_word, "CNY")
        scale = _CN_UNIT_SCALE.get(unit_word, Decimal("1"))
        return {"currency": currency, "unit": unit_word, "scale": scale,
                "row_idx": i, "raw": row_text}
    return None

def _concat_headers(h1: list[str], h2: list[str]) -> list[str]:
    # 逐列拼接两行表头（消除"未分配利润"+"(累计亏损)"被拆行）
    out = []
    n = max(len(h1), len(h2))
    for i in range(n):
        a = _to_halfwidth(h1[i]) if i < len(h1) else ""
        b = _to_halfwidth(h2[i]) if i < len(h2) else ""
        if not a: out.append(b)
        elif not b: out.append(a)
        else:
            # 合并括号片段
            if (a.endswith("(") or a.endswith("（")) and (b.endswith(")") or b.endswith("）")):
                out.append(a + b)
            elif (b.startswith("(") or b.startswith("（")) and not a.endswith(")"):
                out.append(a + b)
            else:
                # 用空格拼，随后再做规范化
                out.append((a + " " + b).strip())
    # 进一步规范常见列名
    out = [re.sub(r"\s+", "", x) for x in out]  # 删除列名中多余空格
    # 特例：把"未分配利润 (累计亏损)"等合成"未分配利润(累计亏损)"
    out = [x.replace("未分配利润(累计亏损)", "未分配利润(累计亏损)") for x in out]
    return out

def _maybe_merge_multirow_header(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    若顶部存在两行/三行表头，尝试合并；返回 (df_no_header_rows, meta)
    meta 里记录 merged_header 列名数组与被删除的表头行数。
    """
    meta = {"header_rows_removed": 0, "merged_header": None}
    if df.empty:
        return df, meta

    # 预清洗顶部 3 行
    top = [df.iloc[i].astype(str).tolist() for i in range(min(3, len(df)))]
    is_hdr = [_looks_like_header_row(r) for r in top]

    # 情况 A：第1&第2行都是表头 → 合并
    if len(top) >= 2 and is_hdr[0] and is_hdr[1]:
        merged = _concat_headers(top[0], top[1])
        df2 = df.iloc[2:].reset_index(drop=True)
        df2.columns = pd.Index(merged[:len(df2.columns)].copy())
        meta.update({"header_rows_removed": 2, "merged_header": merged})
        return df2, meta

    # 情况 B：只有第1行为表头 → 设为列名
    if is_hdr[0]:
        hdr = [_to_halfwidth(x) for x in top[0]]
        df2 = df.iloc[1:].reset_index(drop=True)
        df2.columns = pd.Index([re.sub(r"\s+", "", x) for x in hdr[:len(df2.columns)]])
        meta.update({"header_rows_removed": 1, "merged_header": list(df2.columns)})
        return df2, meta

    return df, meta

def _to_number(x: str, scale: Decimal) -> str | float:
    """
    将字符串数值标准化：
      - 去千分位（逗号）
      - (x) 或 （x） 转负数
      - 全角标点转半角
      - 按单位缩放（万元→乘 10000）
    返回 float（或原串若确实不是数）。
    """
    s = _to_halfwidth(x)
    if not s:
        return s
    # 纯百分比保留原样（可按需要扩展）
    if s.endswith("%"):
        try:
            core = s[:-1].replace(",", "")
            val = Decimal(core)
            return float(val)  # 也可返回百分比小数：float(val)/100
        except Exception:
            return s

    neg = False
    # 括号负数
    if (s.startswith("(") and s.endswith(")")) or (s.startswith("（") and s.endswith("）")):
        neg = True
        s = s[1:-1]
    # 去千分位等字符
    s = s.replace(",", "").replace("，", "").replace(" ", "")
    # 特殊破折号/空值
    if s in {"-", "—", "–", "— —", "--", "N/A", "NA"}:
        return ""

    try:
        val = Decimal(s)
        if neg:
            val = -val
        if scale and scale != 1:
            val = val * scale
        return float(val)
    except (InvalidOperation, ValueError):
        return x  # 非数值，原样返回

def _clean_table_df(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    对 Camelot 产出的 DataFrame 做清洗：
      1) 全角/空格统一
      2) 单位/币种检测并"上卷"（记录 meta，并在列名附加 [人民币元]）
      3) 合并多行表头（如"未分配利润(累计亏损)"）
      4) 数值标准化（去千分位、括号负数、按单位缩放）
    返回 (df_clean, meta)
    """
    # 统一字符串
    df = df_raw.copy()
    df = df.map(lambda x: _to_halfwidth(x))

    # 1) 单位/币种检测（可能占据第一行）
    unit_info = _detect_unit_and_currency(df)
    if unit_info:
        rm = unit_info["row_idx"]
        # 去掉单位行
        df = df.drop(index=rm).reset_index(drop=True)

    # 2) 合并多行表头
    df, hdr_meta = _maybe_merge_multirow_header(df)

    # 3) 列名后缀加单位（上卷到表头）
    col_suffix = ""
    if unit_info:
        cur = unit_info["currency"] or "CNY"
        unit = unit_info["unit"] or "元"
        col_suffix = f"[{ '人民币' if cur=='CNY' else cur }{unit}]"
        # 仅对明显"金额列"追加后缀（含"金额/余额/数/合计/利润/资产/负债"等关键词）
        new_cols = []
        for c in df.columns:
            c2 = str(c)
            if re.search(r"(金额|余额|数|合计|利润|资产|负债|净额|成本|费用|现金|收入|支出|项目|本期|期末|期初)", c2):
                new_cols.append(c2 + col_suffix)
            else:
                new_cols.append(c2)
        df.columns = new_cols

    # 4) 数值标准化
    scale = unit_info["scale"] if unit_info else Decimal("1")
    
    # 首先识别哪些列主要包含数值数据
    numeric_columns = set()
    for col_idx in range(len(df.columns)):
        col_values = df.iloc[:, col_idx].astype(str).tolist()
        numeric_count = 0
        total_non_empty = 0
        
        for val in col_values:
            val_clean = _to_halfwidth(str(val)).strip()
            if not val_clean or val_clean in {"-", "—", "–", "— —", "--", "N/A", "NA"}:
                continue
            total_non_empty += 1
            # 检查是否包含数字、千分位逗号、括号等数值特征
            if re.search(r"[0-9\(\),，．.％%]", val_clean):
                numeric_count += 1
        
        # 如果超过50%的非空值包含数值特征，则认为该列为数值列
        if total_non_empty > 0 and numeric_count / total_non_empty > 0.5:
            numeric_columns.add(col_idx)
    
    def _maybe_numeric(v, col_idx=None):
        if v is None:
            return v
        s = str(v)
        
        # 如果该列被识别为数值列，优先尝试数值转换
        if col_idx is not None and col_idx in numeric_columns:
            # 对于数值列，即使不包含明显数值特征也尝试转换（可能是空白、文本等）
            converted = _to_number(s, scale)
            # 如果转换后不是原字符串，说明成功转换为数值
            if converted != s:
                return converted
            # 否则至少进行半角标准化
            return _to_halfwidth(s)
        
        # 非数值列的原有逻辑：若字符串里含数字或括号/逗号，尝试转数
        if re.search(r"[0-9\(\),，．.％%]", s):
            return _to_number(s, scale)
        return _to_halfwidth(s)

    # 按列应用数值转换，传入列索引信息
    for col_idx, col_name in enumerate(df.columns):
        df[col_name] = df[col_name].apply(lambda x: _maybe_numeric(x, col_idx))

    meta = {
        "unit": unit_info["unit"] if unit_info else None,
        "currency": unit_info["currency"] if unit_info else None,
        "scale_applied": str(scale) if unit_info else "1",
        "header_rows_removed": hdr_meta["header_rows_removed"],
        "merged_header": hdr_meta["merged_header"],
        "notes": "values normalized: thousands removed, parentheses as negative, full-width normalized",
    }
    return df, meta

# === end of helpers ===========================================================


def extract_text(path: Path) -> Dict[str, Any]:
    """Extract text from a PDF as Markdown or plain text.

    If PyMuPDF4LLM is installed, this function uses its
    :func:`pymupdf4llm.to_markdown` utility to convert pages to
    GitHub‑compatible Markdown【401008996391982†L116-L151】.  Otherwise it falls back to
    PyMuPDF's ``get_text("text")`` to extract a simple concatenation of
    text blocks.

    Parameters
    ----------
    path : Path
        Path to the PDF file.

    Returns
    -------
    dict
        A dictionary with keys ``"type"`` and ``"content"``.  The
        ``"type"`` is either ``"markdown"`` or ``"plain"``, and
        ``"content"`` is the extracted text.
    """
    if _HAS_PYMUPDF4LLM:
        try:
            md_text = pymupdf4llm.to_markdown(str(path))
            return {"type": "markdown", "content": md_text}
        except Exception as e:
            logger.warning("pymupdf4llm failed for %s: %s", path, e)
    # Fallback to plain text extraction using fitz
    doc = fitz.open(str(path))
    text_pieces: List[str] = []
    for page in doc:
        try:
            page_text = page.get_text("text")
        except Exception as e:
            logger.warning("Failed to extract text from page %d of %s: %s", page.number, path, e)
            continue
        text_pieces.append(page_text)
    doc.close()
    return {"type": "plain", "content": "\n".join(text_pieces)}


def _extract_tables(pdf_path: Path) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    if not _HAS_CAMELOT:
        return tables
    try:
        # 优先 lattice（边框表格）
        tl = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
        for idx, t in enumerate(tl):
            df_raw = t.df
            # ★ 新增：清洗
            df_clean, meta = _clean_table_df(df_raw)
            tables.append({
                "index": idx,
                "flavor": "lattice",
                "shape_raw": list(df_raw.shape),
                "shape_clean": list(df_clean.shape),
                "raw_json": df_raw.to_json(orient="split", force_ascii=False),
                "clean_json": df_clean.to_json(orient="split", force_ascii=False),
                "meta": meta,
            })
        return tables
    except Exception as e:
        # lattice 失败不阻断，转 stream
        pass
    try:
        ts = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
        for idx, t in enumerate(ts):
            df_raw = t.df
            # ★ 新增：清洗
            df_clean, meta = _clean_table_df(df_raw)
            tables.append({
                "index": idx,
                "flavor": "stream",
                "shape_raw": list(df_raw.shape),
                "shape_clean": list(df_clean.shape),
                "raw_json": df_raw.to_json(orient="split", force_ascii=False),
                "clean_json": df_clean.to_json(orient="split", force_ascii=False),
                "meta": meta,
            })
    except Exception:
        return tables
    return tables


def extract_tables(path: Path) -> List[Dict[str, Any]]:
    """Extract tables from a PDF using Camelot with cleaning and normalization.

    This function uses the new _extract_tables implementation that:
    - Attempts lattice method first, then falls back to stream
    - Applies data cleaning and normalization via _clean_table_df
    - Returns both raw and cleaned JSON data with metadata
    - Detects units, currencies, and applies proper scaling
    - Merges multi-row headers and normalizes numeric values

    Parameters
    ----------
    path : Path
        Path to the PDF file.

    Returns
    -------
    list of dict
        A list of tables with both raw_json and clean_json data plus metadata.
    """
    return _extract_tables(path)


def extract_images(path: Path, images_dir: Path) -> List[Dict[str, Any]]:
    """Extract embedded images from a PDF and save them to disk.

    For each image found in the document, this function writes the image
    data to a file in ``images_dir``.  Metadata about the image (page
    number, index, file name, width, height) is returned in a list of
    dictionaries.  If ``images_dir`` does not exist it will be created.

    Parameters
    ----------
    path : Path
        Path to the PDF file.
    images_dir : Path
        Directory where extracted images should be stored.

    Returns
    -------
    list of dict
        A list of dictionaries containing metadata about each saved image.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    doc = fitz.open(str(path))
    for page_index, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception as e:
                logger.warning("Failed to load image xref %d on page %d: %s", xref, page_index + 1, e)
                continue
            # Skip images that are embedded as masks or invalid
            if pix.n < 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_name = f"page{page_index+1}_img{img_index+1}.png"
            img_path = images_dir / img_name
            pix.save(str(img_path))
            results.append(
                {
                    "page": page_index + 1,
                    "index": img_index + 1,
                    "file_name": img_name,
                    "width": pix.width,
                    "height": pix.height,
                }
            )
            pix = None  # free resources
    doc.close()
    return results


def _ocr_pages_with_ppstructure(pdf_path: Path, tmp_img_dir: Path) -> Dict[str, Any]:
    """将每页渲染为 PNG，调用 PP-Structure 做表格/文本识别，返回 {pages:[...]}"""
    results = {"pages": []}
    tmp_img_dir.mkdir(parents=True, exist_ok=True)
    ocr = PPStructure(show_log=False, layout=False)  # 只跑表格结构；可按需开启 layout
    with fitz.open(str(pdf_path)) as doc:
        for pno, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x 放大提升OCR效果
            img_path = tmp_img_dir / f"pp_{pno:04d}.png"
            pix.save(img_path)
            ocr_res = ocr(str(img_path))
            # 结构化清洗（保存表格/文本块，保留坐标/置信度）
            page_items = []
            for item in ocr_res:
                keep = {
                    "type": item.get("type"),
                    "bbox": item.get("bbox"),
                    "score": item.get("score"),
                }
                if "res" in item:
                    # 文本块
                    if isinstance(item["res"], list):
                        keep["text"] = " ".join(seg.get("text","") for seg in item["res"])
                    # 表格结构（html）
                    if isinstance(item["res"], dict) and "html" in item["res"]:
                        keep["table_html"] = item["res"]["html"]
                page_items.append(keep)
            results["pages"].append({"page": pno, "items": page_items})
    return results


def parse_pdf(
    path: Path,
    output_dir: Path,
    *,
    enable_ocr: bool = False,
    ocr_all_pages: bool = False,
) -> Dict[str, Any]:
    """Parse a single PDF file into structured data.

    This function extracts text, tables and images from ``path`` and
    returns a dictionary with the results.  Images are exported into
    ``output_dir/images/{stem}/`` where ``stem`` is the base filename
    without extension.

    Parameters
    ----------
    path : Path
        Path to the PDF file to parse.
    output_dir : Path
        Base directory where images will be stored.  A subdirectory
        named ``images/{stem}`` will be created.

    Returns
    -------
    dict
        A dictionary containing keys ``file_name``, ``text``,
        ``tables`` and ``images``.
    """
    file_name = path.name
    stem = path.stem
    # Extract text (markdown or plain)
    text_info = extract_text(path)
    # Extract tables
    tables = extract_tables(path)
    # Extract images into a subfolder named after the file
    images_subdir = output_dir / "images" / stem
    images = extract_images(path, images_subdir)
    result = {
        "file_name": file_name,
        "text": text_info,
        "tables": tables,
        "images": images,
    }

    # 自动 OCR 回退：文本很少（可能是扫描件）或没有表格且启用OCR
    need_ocr = enable_ocr and (ocr_all_pages or (not result["tables"] or (result["text"]["type"]=="plain" and len(result["text"].get("content","").strip())<30)))
    if need_ocr and _HAS_PPOCR:
        ocr_tmp = Path(output_dir) / "ocr_tmp" / path.stem
        try:
            ocr_res = _ocr_pages_with_ppstructure(path, ocr_tmp)
            result["ocr"] = ocr_res  # 附加 OCR 结果（不覆盖原始 text/tables）
        except Exception as e:
            result["ocr_error"] = f"{type(e).__name__}: {e}"
    return result


async def _parse_worker(
    semaphore: asyncio.Semaphore,
    file_path: Path,
    output_dir: Path,
    enable_ocr: bool = False,
    ocr_all_pages: bool = False,
) -> Tuple[str, bool, Dict[str, Any]]:
    """Internal worker to parse a PDF within a semaphore and thread pool.

    Returns a tuple of ``(file_name, success, data_or_error)`` where the
    third element is the parsed data on success or an error message on
    failure.
    """
    async with semaphore:
        loop = asyncio.get_running_loop()
        try:
            # Offload the CPU/IO intensive parsing to a separate thread
            result = await loop.run_in_executor(None, lambda: parse_pdf(file_path, output_dir, enable_ocr=enable_ocr, ocr_all_pages=ocr_all_pages))
            return (file_path.name, True, result)
        except Exception as exc:
            logger.error("Failed to parse %s: %s", file_path, exc)
            return (file_path.name, False, {"error": str(exc)})


async def parse_pdfs(
    file_paths: Sequence[str],
    output_dir: str,
    max_concurrency: int = 4,
    *,
    enable_ocr: bool = False,
    ocr_all_pages: bool = False,
) -> List[Tuple[str, bool, Dict[str, Any]]]:
    """Parse multiple PDF files concurrently.

    Parameters
    ----------
    file_paths : Sequence[str]
        List of paths to PDF files to be processed.
    output_dir : str
        Directory where extracted images will be stored and where JSON
        outputs may be written.  The directory will be created if it
        doesn't exist.
    max_concurrency : int, optional
        Maximum number of concurrent parsing tasks.

    Returns
    -------
    list of tuple
        A list of ``(file_name, success, data_or_error)`` tuples.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []
    for fp in file_paths:
        tasks.append(
            _parse_worker(
                semaphore,
                Path(fp),
                out_dir,
                enable_ocr,
                ocr_all_pages,
            )
        )
    return await asyncio.gather(*tasks)


__all__ = ["parse_pdf", "parse_pdfs"]