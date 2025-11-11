# Stock Market Explorer (Streamlit frontend)

This repository contains a Streamlit frontend (`app.py`) that implements EDA and sentiment analysis flows based on `ntcc_project.ipynb`.

Quick start

1. Create a Python environment (recommended).
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
streamlit run app.py
```

Notes

- If you have the notebook's produced `stock_data.csv` in the project root, the app will try to load it automatically when no files are uploaded.
- The app performs sentiment using NLTK VADER if available; otherwise it falls back to TextBlob when installed.
- The prediction demo is a tiny Ridge regression using recent lagged closes — it's illustrative and fast compared to training an LSTM.
