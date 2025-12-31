import json
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from utils.metric import compute_metrics
from utils.agents import PortfolioAllocation

class TradingSimulator:
    def __init__(self, trading_agent, account, tickers):
        self.trading_agent = trading_agent
        self.account = account
        self.tickers = tickers
        self.trading_log = []
        self.log_filename = "trading_log.json"

    def load_previous_state(self, filename="trading_log.json"):
        """Load the last portfolio state from a previous trading log"""
        try:
            with open(filename, 'r') as f:
                log_data = json.load(f)

            if log_data.get("daily_logs"):
                last_day = log_data["daily_logs"][-1]

                # Restore cash balance
                self.account.cash_balance = last_day["cash"]

                # Restore positions
                self.account.positions = last_day["positions"].copy()

                # Restore prices
                self.account.prices = last_day["prices"].copy()

                # Restore equity history
                for daily_log in log_data["daily_logs"]:
                    date_df = pd.to_datetime(daily_log["date"])
                    self.account.equity.loc[date_df, "equity"] = daily_log["total_equity"]

                # Restore previous trading log
                self.trading_log = log_data["daily_logs"].copy()

                print(f"\n✅ Loaded previous state from {filename}")
                print(f"   Last trading date: {last_day['date']}")
                print(f"   Cash: ${last_day['cash']:,.2f}")
                print(f"   Total equity: ${last_day['total_equity']:,.2f}")
                print(f"   Positions: {len([k for k, v in last_day['positions'].items() if v > 0])} stocks")

                return True
        except FileNotFoundError:
            print(f"\n⚠️ Previous log file '{filename}' not found. Starting fresh.")
            return False
        except Exception as e:
            print(f"\n⚠️ Error loading previous state: {e}")
            return False

    def get_stock_prices(self, date: str):
        prices = {}
        next_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=date, end=next_date, progress=False)
                if not df.empty and 'Close' in df.columns:
                    prices[ticker] = float(df['Close'].iloc[-1])
                else:
                    return None
            except:
                return None

        return prices

    async def step(self, date: str):
        stock_prices = self.get_stock_prices(date)

        if stock_prices is None:
            print(f"Market closed on {date}. Skipping.")
            return

        self.account.update_prices(stock_prices)
        
        input = {"messages": [{"role": "user", "content": f"Evaluation date is {date}."}]}
        
        response = self.trading_agent.invoke(
            input = input
        )

        response_content = response["messages"][-1].content
        print(response_content)

        # Parse JSON string to PortfolioAllocation Pydantic model
        allocation_data = json.loads(response_content)
        portfolio_allocation = PortfolioAllocation(**allocation_data)

        stock_allocations = portfolio_allocation.allocations

        prev_equity = self.account.get_total_value()
        cur_equity = self.account.apply_allocation(stock_allocations)
        date_df = pd.to_datetime(date)
        self.account.equity.loc[date_df, "equity"] = cur_equity

        equity_df = self.account.equity
        metrics = compute_metrics(equity_df)
        cr = metrics["CR"]
        sr = metrics["SR"]
        mdd = metrics["MDD"]
        wr = metrics["WR"]
        vol = metrics["Vol"]
        print(f"CR: {cr:,.2f}  SR: {sr:,.2f}  MDD: {mdd:,.2f}  WR: {wr:,.2f}  Vol: {vol:,.2f}")

        daily_return = (cur_equity - prev_equity) / prev_equity if prev_equity > 0 else 0

        log_entry = {
            "date": date,
            "allocations": {k: round(v, 4) for k, v in stock_allocations.items()},
            "positions": {k: round(v, 4) for k, v in self.account.positions.items()},
            "prices": {k: round(v, 2) for k, v in stock_prices.items()},
            "cash": round(self.account.cash_balance, 2),
            "total_equity": round(cur_equity, 2),
            "daily_return": round(daily_return * 100, 4),
            "metrics": {
                "cumulative_return": round(cr, 2),
                "sharpe_ratio": round(sr, 4),
                "max_drawdown": round(mdd, 2),
                "win_rate": round(wr, 2),
                "volatility": round(vol, 2)
            }
        }

        self.trading_log.append(log_entry)
        self.save_log_to_json(self.log_filename)

    async def run(self, start_date: str, end_date: str):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        initial_value = self.account.cash_balance

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            print(f"\n===== {date_str} =====")
            await self.step(date_str)
            current += timedelta(days=1)

        final_value = self.account.get_total_value()
        profit = final_value - initial_value

        print("\n📊 === Simulation Summary ===")
        print(f"Initial Cash: ${initial_value:,.2f}")
        print(f"Final Value:  ${final_value:,.2f}")
        print(f"Profit:       ${profit:,.2f}")

        self.save_log_to_json()

    def save_log_to_json(self, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_log.json"

        log_data = {
            "simulation_info": {
                "start_date": self.trading_log[0]["date"] if self.trading_log else None,
                "end_date": self.trading_log[-1]["date"] if self.trading_log else None,
                "total_days": len(self.trading_log),
                "tickers": self.tickers
            },
            "final_metrics": self.trading_log[-1]["metrics"] if self.trading_log else {},
            "final_portfolio": {
                "cash": self.trading_log[-1]["cash"] if self.trading_log else 0,
                "positions": self.trading_log[-1]["positions"] if self.trading_log else {},
                "total_equity": self.trading_log[-1]["total_equity"] if self.trading_log else 0
            },
            "daily_logs": self.trading_log
        }

        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n💾 Trading log saved to: {filename}")