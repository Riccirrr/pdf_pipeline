# pdf_pipeline
## install dependency
chmod +x install.sh
./install.sh

## usage
python -m pdf_pipeline.pipeline \
  --urls urls.txt \
  --download-dir ./downloads \
  --output-dir ./outputs \
  --download-concurrency 24 \
  --parse-concurrency 8 \
  --enable-ocr