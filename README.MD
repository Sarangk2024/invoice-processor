# 📑 Invoice & PO Matching Tool

An intelligent document processing application built with Streamlit and Google's Gemini API to automate the two-way and three-way matching process for Accounts Payable (AP). It extracts key data from uploaded PDF Invoices and Purchase Orders, aggregates line items, and provides a clear, color-coded match/mismatch summary.

## ✨ Features

* **Intelligent Data Extraction (Gemini-Pro):** Uses the Gemini API to analyze PDF text and image content, intelligently extracting key fields like Invoice Number, PO Number, Dates, Vendor Name, and Line Items.
* **Multi-Invoice Handling:** The model is prompted to handle and aggregate data from multiple invoices contained within a single text block.
* **Item Aggregation & Normalization:** Automatically sums quantities and total prices for line items with matching descriptions across documents.
* **2-Way & 3-Way Matching Logic:** Compares critical fields:
    * **PO Number:** Invoice PO vs. PO Document PO.
    * **Vendor Name:** Fuzzy matching to account for minor discrepancies.
    * **Total Amount:** Strict comparison.
    * **Line Items & Quantities:** Checks for quantity and price overages from the Invoice compared to the PO.
* **Intuitive UI (Streamlit):** Clean, custom-styled interface for uploading documents and viewing results side-by-side.
* **Status Indicators:** Provides a clear `APPROVED ✅` or `NEEDS REVIEW ⚠️` status based on the matching results.

## 🛠️ Prerequisites

Before running the application, ensure you have the following installed:

1.  **Python 3.8+**
2.  **Google AI API Key:** You must have a Gemini API key.
3.  **Tesseract OCR:** Required for reading image-based PDFs (scanned documents).
    * **Windows:** Install Tesseract from the [official GitHub page](https://github.com/tesseract-ocr/tesseract) and update the `pytesseract.pytesseract.tesseract_cmd` path in the code if it's not in your system PATH.
    * **Linux/Mac:** Usually installed via package managers (`sudo apt install tesseract-ocr` or `brew install tesseract`).

## 🚀 Setup and Installation

### 1. Clone the repository (or save the code)

Save the provided Python code as `app.py`.

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
### 3. Adding API key
```bash
create a .env file and paste your gemini api key like this

GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash 
streamlit run app.py
```
