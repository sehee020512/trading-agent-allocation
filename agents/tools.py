from langchain.tools import tool
from langchain_experimental.tools import PythonREPLTool

import yfinance as yf
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
import requests
import feedparser
from urllib.parse import quote_plus

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

### Code Interpreter ###
code_interpreter = PythonREPLTool()

### Getting Price ###
@tool
def get_price(tickers: str, start_date: str, end_date: str) -> str:
    """
    Retrieve historical daily closing prices for stock tickers within a specified date range.
    Returns a DataFrame with Date as index and each ticker as a column.

    - tickers: comma-separated string, e.g. "AAPL,MSFT"
    - start_date: YYYY-MM-DD (inclusive)
    - end_date: YYYY-MM-DD (inclusive)
    """

    ticker_list = [t.strip() for t in tickers.split(",")]

    df = yf.download(
        ticker_list,
        start=start_date,
        end=(datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
    )
    if "Close" in df.columns:
        close_prices = df["Close"]
        
        if isinstance(close_prices, pd.Series):
            close_prices = close_prices.to_frame(name=ticker_list[0])
    else:
        return "Error: 'Close' column not found in data"

    close_prices = close_prices.round(4)

    return close_prices.to_csv()

### Searching News ###
@tool
def news_searcher(
    query: str,
    start_date: str,
    end_date: str,
    max_results: int = 5
) -> str:
    """
    Fetch news article titles from Google News RSS feed for a given query and date range.
    Returns only the titles of news articles as a formatted string.

    - query: search keywords (e.g., "AAPL earnings", "Federal Reserve")
    - start_date: YYYY-MM-DD format (inclusive)
    - end_date: YYYY-MM-DD format (inclusive)
    - max_results: maximum number of titles to return (default: 5)
    """

    print(f"  - Fetching news titles for '{query}' from {start_date} to {end_date}")

    rss_query = quote_plus(f"{query} after:{start_date} before:{end_date}")
    url = f"https://news.google.com/rss/search?q={rss_query}&hl=en-US&gl=US&ceid=US:en"

    try:
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
    except Exception as e:
        return f"Error fetching news: {str(e)}"

    if not feed.entries:
        return f"No news found for query '{query}' in the specified date range."

    titles = []
    for i, entry in enumerate(feed.entries[:max_results]):
        try:
            title = entry.title.strip()
            titles.append(f"{i+1}. {title}")
        except Exception:
            continue

    if not titles:
        return f"No valid news titles found for query '{query}'."

    result = f"News titles for '{query}' ({start_date} to {end_date}):\n\n"
    result += "\n".join(titles)

    return result