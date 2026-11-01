# Scanned PDF OCR

NitiSaathi uses local Tesseract OCR as a fallback for PDF pages that have no usable text layer. Text PDFs continue through the existing PyMuPDF extraction path. OCR text is stored per original page, then follows the same IndicLID, chunking, vector indexing, Q&A, and citation flow. This adds no per-page cloud OCR charge and does not send the PDF to a new provider.

## Local setup

Install the Tesseract OCR engine on the backend machine. On Windows, the [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) provides Tesseract builds. Then run `scripts/install_ocr_language_data.ps1` to download the small, fast language models for English and the eight Indic speech languages currently enabled in Bulbul v4. The model data is kept under ignored `data/ocr/tessdata/`, not in Git. Additional OCR languages require a code mapping and a model file before use. Run `tesseract --list-langs` to check system-installed packs. PyMuPDF invokes Tesseract directly; a Python wrapper package is not needed. See the [PyMuPDF Tesseract language pack setup](https://pymupdf.readthedocs.io/en/latest/ocr/tesseract-language-packs.html).

If PyMuPDF cannot find the language-data folder, set `TESSDATA_PREFIX` in the backend environment (an example is in `.env.example`).

The backend currently maps English, Hindi, Bengali, Gujarati, Kannada, Marathi, Punjabi, Tamil, and Telugu to Tesseract packs. This matches the nine language codes enabled by the current Bulbul v4 backend configuration. Do not advertise OCR support for other languages until their code mapping and model data are added and tested.

## Upload contract

`POST /documents/upload` accepts multipart form data:

- `file`: a PDF (existing 20 MB backend limit)
- `document_language`: the source document language, such as `en-IN`, `hi-IN`, `ta-IN`, or `bn-IN`. It defaults to `en-IN` for existing clients. Mixed English/Hindi pages can use `en-IN+hi-IN`.

The frontend should send the language in which the PDF is written. This is an OCR hint; IndicLID still detects languages from the extracted chunks afterward.

The upload response includes `ocr_pages`, and each page preview includes `extraction_method` (`native`, `tesseract_ocr`, or `unreadable`). Unreadable pages are reported in `warnings` and excluded from retrieval. If no page yields readable text, upload returns an error instead of indexing an empty document.

## OCR flow

1. PyMuPDF extracts existing text page by page.
2. Pages with no text, very short text over an image, or a document with no useful native text are selected for OCR.
3. PyMuPDF runs Tesseract at 300 DPI using the selected language pack.
4. Extracted text replaces the selected page text and proceeds through the existing backend pipeline.

The original PDF remains the citation source. OCR supplies page-level text, not character coordinates, so citations identify the supporting page and passage in the extracted text; they do not highlight a bounding box on the image.

## Known boundaries

- The backend upload limit remains 20 MB.
- OCR accuracy depends on scan quality and the correct `document_language` hint.
- A matching Tesseract language pack must be installed; missing engine/data errors are returned clearly.
- A PDF can contain a readable text layer mixed with scanned pages; likely scanned pages use OCR while digital pages stay on the native parser.
