import asyncio
import pandas as pd
import os
from utils.account import StockAccount
from utils.simulator import TradingSimulator
from utils.agents import trading_agent
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


async def main():

    ###
    initial_cash = 1000
    start_date = "2025-09-01"
    end_date = "2025-09-30"
    ###

    account = StockAccount(cash_balance=initial_cash)
    sim = TradingSimulator(
        trading_agent,
        account,
        ["AAPL", "MSFT", "NVDA", "JPM", "V", "JNJ", "UNH", "PG", "KO", "XOM", "CAT", "WMT", "META", "TSLA", "AMZN"]
    )

    # Load previous portfolio state if available
    sim.load_previous_state("trading_log.json")

    await sim.run(start_date, end_date)


if __name__ == "__main__":
    asyncio.run(main())