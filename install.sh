#!/usr/bin/env bash
set -euo pipefail

say()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[1;31m[ERR ]\033[0m %s\n" "$*" >&2; }

# 0) 先检查 uv
if ! command -v uv >/dev/null 2>&1; then
  err "未检测到 'uv'。请先安装：curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
say "Python: $(python -V 2>/dev/null || echo 'unknown')"
say "Using uv: $(uv --version 2>/dev/null || echo 'unknown')"

# 1) Ghostscript：安装/定位 + 软链 + 环境变量
GS_BIN=""
GS_LIB=""
if command -v brew >/dev/null 2>&1; then
  if ! brew list gs >/dev/null 2>&1 && ! brew list ghostscript >/dev/null 2>&1; then
    say "安装 Ghostscript（via Homebrew）..."
    brew install ghostscript
  else
    say "检测到 Ghostscript 已安装（Homebrew）"
  fi
  GS_PREFIX="$(brew --prefix gs 2>/dev/null || brew --prefix ghostscript 2>/dev/null || true)"
  if [[ -n "${GS_PREFIX}" ]]; then
    GS_BIN="${GS_PREFIX}/bin/gs"
    GS_LIB="${GS_PREFIX}/lib/libgs.dylib"
  fi
fi
# 补充：若无 brew，直接 which gs
if [[ -z "${GS_BIN}" ]]; then
  GS_BIN="$(command -v gs || true)"
fi

# PATH / 软链 / 动态库回退路径
if [[ -d "/opt/homebrew/bin" ]]; then
  export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"
fi
if [[ -n "${GS_LIB:-}" && -f "${GS_LIB}" ]]; then
  mkdir -p "${HOME}/lib"
  ln -sf "${GS_LIB}" "${HOME}/lib/libgs.dylib"
  say "已在 ~/lib 链接 libgs.dylib -> ${GS_LIB}"
fi
if [[ -n "${GS_BIN:-}" ]]; then
  export CAMELOT_GS="${CAMELOT_GS:-${GS_BIN}}"
fi
export DYLD_FALLBACK_LIBRARY_PATH="${HOME}/lib:/opt/homebrew/lib:/usr/local/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"

say "Ghostscript 可执行: ${GS_BIN:-not-found}"
say "CAMELOT_GS: ${CAMELOT_GS:-not-set}"
say "DYLD_FALLBACK_LIBRARY_PATH: ${DYLD_FALLBACK_LIBRARY_PATH}"

# 2) 分步安装（全部禁缓存 + 刷新），按你给的顺序
say "[1/5] 安装打包工具（Paddle 运行需要 setuptools）"
uv pip install --refresh --no-cache "setuptools>=68,<75" "wheel>=0.43"

say "[2/5] 安装大包（numpy/pandas/PyMuPDF）"
uv pip install --refresh --no-cache "numpy==1.26.4"
uv pip install --refresh --no-cache "pandas==2.2.2"
uv pip install --refresh --no-cache "PyMuPDF==1.24.10"

say "[3/5] 安装其余核心 + Camelot"
uv pip install --refresh --no-cache "aiohttp==3.9.5" "tqdm==4.66.4" "pdfminer.six==20231228" "camelot-py==0.11.0"
uv pip install --refresh --no-cache "opencv-python-headless==4.9.0.80"
uv pip install --refresh --no-cache "ghostscript==0.7"   # Python 绑定

say "[4/5] 安装 OCR 栈"
uv pip install --refresh --no-cache "shapely==2.0.2" "scipy==1.11.4" "scikit-image==0.21.0" "pillow>=10,<11"
uv pip install --refresh --no-cache "paddlepaddle==2.6.1"
uv pip install --refresh --no-cache --no-deps "paddleocr==2.7.0.3"

say "[5/5] 自检：版本与 Camelot lattice 快测"
python - <<'PY'
import os, shutil, importlib, glob
def v(m):
    try:
        mod=importlib.import_module(m)
        return getattr(mod,"__version__","OK")
    except Exception as e:
        return f"ERR: {type(e).__name__}: {e}"
mods = ["numpy","pandas","fitz","aiohttp","camelot","cv2","pdfminer","ghostscript","paddle","paddleocr","setuptools"]
for m in mods:
    print(f"{m:12s} =", v(m))
print("CAMELOT_GS      =", os.environ.get("CAMELOT_GS"))
print("which gs        =", shutil.which("gs"))
# 试跑 lattice：如 downloads/ 下有 PDF 就测第一页
try:
    import camelot
    pdfs = sorted(glob.glob("downloads/*.pdf")) + sorted(glob.glob("downloads/*.PDF"))
    if pdfs:
        t = camelot.read_pdf(pdfs[0], pages="1", flavor="lattice")
        print("lattice tables on page 1:", len(t))
    else:
        print("未发现 downloads/*.pdf，跳过 lattice 试跑")
except Exception as e:
    print("camelot lattice test FAIL:", repr(e))
PY

say "✅ 安装与环境配置完成。若仍提示 Ghostscript 未安装："
say "   1) 确认 CAMELOT_GS：      echo \$CAMELOT_GS"
say "   2) 确认 libgs 软链：       ls -l ~/lib/libgs.dylib"
say "   3) 手动验证 GS：           gs --version"
