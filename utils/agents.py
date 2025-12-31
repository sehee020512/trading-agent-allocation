from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_experimental.tools import PythonREPLTool
from pydantic import BaseModel, Field, field_validator

from openai.types.responses.web_search_tool import Filters
import yfinance as yf
import pandas as pd
from pandas import DataFrame
from datetime import datetime, timedelta
import requests
import feedparser
from urllib.parse import quote_plus
from typing import List, Dict, Any, Optional

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@tool
def get_price(tickers: str, start_date: str, end_date: str) -> str:
    """
    Retrieve historical daily closing prices for stock tickers within a specified date range.
    Returns a DataFrame with Date as index and each ticker as a column.

    - tickers: comma-separated string, e.g. "AAPL,MSFT"
    - start_date: YYYY-MM-DD (inclusive)
    - end_date: YYYY-MM-DD (inclusive; internally adjusted by +1 day)
    """

    ticker_list = [t.strip() for t in tickers.split(",")]

    df = yf.download(
        ticker_list,
        start=start_date,
        end=(datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
    )

    if len(ticker_list) == 1:
        close_prices = pd.DataFrame({ticker_list[0]: df["Close"]})
    else:
        close_prices = df["Close"]

    close_prices = close_prices.round(4)

    return close_prices.to_csv()

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

code_interpreter = PythonREPLTool()

instruction = """
You are a professional portfolio manager analyzing market information to produce portfolio allocations.

Portfolio Objective:
- Maximize expected return per unit of risk
- Aim to outperform a reasonable baseline over a 1-3 month horizon
- Use CASH tactically to manage downside risk during unfavorable conditions

Evaluation Criteria:
- Favor allocations that improve expected excess return and risk-adjusted performance
- Maintain diversification across sectors, styles, and market capitalizations
- Be mindful of turnover, liquidity, and position concentration

Portfolio Construction Principles:
- Diversify across multiple assets and sectors
- Consider momentum, market trends, and relevant fundamentals
- Balance growth and value exposures
- Maintain appropriate position sizing
- Total allocation should sum to 1.0
- CASH is a valid and allocatable asset

Available Assets:
AAPL, MSFT, NVDA, JPM, V, JNJ, UNH, PG, KO, XOM, CAT, WMT, META, TSLA, AMZN, CASH

Tool Usage Policy:
- Use get_price for all quantitative market data, including historical prices, returns, volatility, and trend analysis.
- Use news_searcher for qualitative and contextual information such as macroeconomic news, earnings reports, sector trends, and market sentiment.
- Use code_interpreter to perform calculations, data aggregation, statistical analysis, or transformations based on data obtained from tools.

Constraints:
- CASH allocation should be justified by market conditions
"""

class PortfolioAllocation(BaseModel):
    reasoning: str = Field(
        description="Brief explanation of why this allocation is expected to improve risk-adjusted returns"
    )
    allocations: Dict[str, float] = Field(
        description="A dictionary where keys are assets and values are their respective "
                    "allocation weights. IMPORTANT: The sum of all values MUST be exactly 1.0. "
                    "Example: {'AAPL': 0.25, 'MSFT': 0.20, 'NVDA': 0.15, 'CASH': 0.40}"
    )

    @field_validator('allocations', mode='after')
    @classmethod
    def ensure_normalization(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if total <= 0:
            raise ValueError("Total allocation sum must be greater than 0")
        
        if abs(total - 1.0) > 0.01:
            return {k: round(float(val) / total, 4) for k, val in v.items()}
        return v

trading_agent = create_agent(
    model = "gpt-5-mini",
    tools = [get_price,
             news_searcher,
             code_interpreter
             ],
    system_prompt = instruction,
    response_format = PortfolioAllocation
)