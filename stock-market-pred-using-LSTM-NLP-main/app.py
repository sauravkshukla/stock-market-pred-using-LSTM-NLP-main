import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Lightweight model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

# Optional sentiment
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.data.find('sentiment/vader_lexicon.zip')
    NLTK_VADER = True
except Exception:
    NLTK_VADER = False

try:
    from textblob import TextBlob
    TEXTBLOB_OK = True
except Exception:
    TEXTBLOB_OK = False

st.set_page_config(page_title="Stock Sentiment Explorer", layout="wide")

st.title("Stock Market Explorer — Frontend from Notebook")

st.markdown("This app recreates the core EDA and sentiment flows from `ntcc_project.ipynb`. Upload price CSV and headlines CSV, or load a pre-prepared `stock_data.csv` if present.")

# Sidebar
st.sidebar.header("Inputs")
price_file = st.sidebar.file_uploader("Upload stock price CSV (Date, Close, Open, High, Low, Volume)", type=['csv'])
headlines_file = st.sidebar.file_uploader("Upload headlines CSV (date or publish_date, headline_text)", type=['csv'])
use_sample = st.sidebar.checkbox("Try to load local 'stock_data.csv' if no upload", value=True)
rolling_window = st.sidebar.slider("Rolling window (days)", min_value=1, max_value=90, value=30)
train_demo = st.sidebar.checkbox("Run a tiny prediction demo (fast)", value=True)

@st.cache_data
def load_csv_from_buffer(buf):
    try:
        return pd.read_csv(buf)
    except Exception:
        return None

# Load data
stock_price = None
stock_headlines = None

if price_file is not None:
    stock_price = load_csv_from_buffer(price_file)
elif use_sample:
    try:
        stock_price = pd.read_csv('stock_data.csv')
        st.sidebar.markdown("Loaded local `stock_data.csv` as price data")
    except Exception:
        stock_price = None

if headlines_file is not None:
    stock_headlines = load_csv_from_buffer(headlines_file)

if stock_price is None and stock_headlines is None:
    st.info("Upload at least the stock price CSV or provide the pre-saved `stock_data.csv` in the app folder.")
    st.stop()

# Processing helpers

def prepare_price_df(df):
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        # try common names
        for c in df.columns:
            if 'date' in c.lower():
                df['Date'] = pd.to_datetime(df[c], errors='coerce')
                break
    df = df.dropna(subset=['Date'])
    df['Date'] = df['Date'].dt.normalize()
    df = df.set_index('Date').sort_index()
    return df


def prepare_headlines_df(df):
    df = df.copy()
    # find date column
    date_col = None
    for c in df.columns:
        if 'date' in c.lower() or 'publish' in c.lower():
            date_col = c
            break
    text_col = None
    for c in df.columns:
        if 'headline' in c.lower() or 'text' in c.lower():
            text_col = c
            break
    if date_col is None or text_col is None:
        return None
    df = df[[date_col, text_col]].drop_duplicates()
    df['publish_date'] = pd.to_datetime(df[date_col].astype(str), errors='coerce')
    df = df.dropna(subset=['publish_date'])
    df['publish_date'] = df['publish_date'].dt.normalize()
    df = df.groupby('publish_date')[text_col].apply(lambda x: ', '.join(x.astype(str))).reset_index()
    df = df.set_index('publish_date').sort_index()
    df.index.name = 'Date'
    df.columns = ['headline_text']
    return df


@st.cache_data
def combine_and_sentiment(price_df, headlines_df=None):
    # Guard: if no price dataframe was provided, return empty DataFrame
    if price_df is None:
        st.error('No price data available to process. Please upload a price CSV or provide a local `stock_data.csv`.')
        return pd.DataFrame()

    p = prepare_price_df(price_df)
    if headlines_df is not None:
        h = prepare_headlines_df(headlines_df)
        if h is None:
            st.warning('Could not detect date/headline columns in headlines file.')
            df = p.copy()
        else:
            df = pd.concat([p, h], axis=1)
    else:
        df = p.copy()

    df = df.dropna(axis=0, how='any')

    # sentiment columns
    df = df.copy()
    df['compound'] = np.nan
    df['neg'] = np.nan
    df['neu'] = np.nan
    df['pos'] = np.nan

    if 'headline_text' in df.columns:
        if NLTK_VADER:
            sid = SentimentIntensityAnalyzer()
            s = df['headline_text'].astype(str).apply(lambda x: sid.polarity_scores(x))
            df['compound'] = s.apply(lambda x: x['compound'])
            df['neg'] = s.apply(lambda x: x['neg'])
            df['neu'] = s.apply(lambda x: x['neu'])
            df['pos'] = s.apply(lambda x: x['pos'])
        elif TEXTBLOB_OK:
            df['polarity'] = df['headline_text'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
            df['subjectivity'] = df['headline_text'].astype(str).apply(lambda x: TextBlob(x).sentiment.subjectivity)
            # map polarity into compound-like scale
            df['compound'] = df['polarity']
            df['neg'] = (df['polarity'] < -0.1).astype(float)
            df['pos'] = (df['polarity'] > 0.1).astype(float)
            df['neu'] = 1 - (df['neg'] + df['pos'])
        else:
            st.warning('No sentiment engine available (install nltk or textblob). Sentiment columns will be empty.')
    return df

# Combine
with st.spinner('Processing data...'):
    stock_data = combine_and_sentiment(stock_price, stock_headlines)

if stock_data.empty:
    st.info('No processed data available. Please upload a valid price CSV (with a Date column) or provide `stock_data.csv` in the app folder.')
    st.stop()

st.header('Data preview')
st.dataframe(stock_data.head(50))

# Basic stats
st.header('Descriptive stats')
st.write(stock_data.describe())

# Time series plot
st.header('Price chart')
fig, ax = plt.subplots(figsize=(10, 4))
if 'Close' in stock_data.columns:
    col = 'Close'
    stock_data[col].plot(ax=ax, label='Close')
else:
    # Try lowercase
    if 'close' in [c.lower() for c in stock_data.columns]:
        col = [c for c in stock_data.columns if c.lower()=='close'][0]
        stock_data[col].plot(ax=ax, label='Close')
    else:
        st.warning('No Close column found to plot price.')
        col = None

# rolling mean
if col is not None:
    stock_data[col].rolling(window=rolling_window).mean().plot(ax=ax, label=f'{rolling_window}-day MA')
ax.set_title('Close Price')
ax.set_xlabel('Date')
ax.legend()
st.pyplot(fig)

# Sentiment time series
if 'compound' in stock_data.columns and not stock_data['compound'].isna().all():
    st.header('Sentiment over time (compound score)')
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    stock_data['compound'].plot(ax=ax2)
    ax2.set_title('Compound Sentiment')
    st.pyplot(fig2)

# Correlation heatmap
st.header('Correlation heatmap')
num_df = stock_data.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
if not num_df.empty:
    corr = num_df.corr()
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)
else:
    st.info('No numeric columns available to compute correlations.')

# Simple prediction demo
if train_demo:
    st.header('Tiny prediction demo (fast, illustrative)')
    # require Close column
    if col is None:
        st.info('No Close column available for prediction demo.')
    else:
        series = stock_data[col].dropna()
        # build lag features
        LAGS = st.slider('Number of lag days to use', 1, 30, 5)
        window = LAGS + 100
        if len(series) < window:
            st.info('Not enough data for demo. Need at least %d rows.' % window)
        else:
            # prepare dataset
            X = []
            y = []
            s = series.values
            for i in range(LAGS, len(s)):
                X.append(s[i-LAGS:i])
                y.append(s[i])
            X = np.array(X)
            y = np.array(y)
            split = int(0.8*len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            # scale
            scaler = MinMaxScaler()
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            X_train_s = scaler.fit_transform(X_train_flat)
            X_test_s = scaler.transform(X_test_flat)
            model = Ridge(alpha=1.0)
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            # show results
            st.write('Test MAE:', float(np.mean(np.abs(preds - y_test))))
            fig4, ax4 = plt.subplots(figsize=(8,4))
            ax4.plot(preds[:100], label='Predicted')
            ax4.plot(y_test[:100], label='Actual')
            ax4.legend()
            st.pyplot(fig4)

st.sidebar.header('Save')
if st.sidebar.button('Export prepared CSV'):
    out = StringIO()
    stock_data.to_csv(out)
    out.seek(0)
    st.download_button('Download prepared CSV', data=out.getvalue(), file_name='stock_data_prepared.csv', mime='text/csv')

st.markdown('---')
st.caption('This Streamlit UI implements a fast, front-end friendly subset of the original notebook: cleaning, sentiment, EDA and a tiny demo model.')
