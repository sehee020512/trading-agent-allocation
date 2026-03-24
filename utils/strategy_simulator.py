import json
import pandas as pd
from datetime import datetime, timedelta
from utils.simulator import TradingSimulator
from agents.trading_agents import StrategyPortfolioAllocation
from agents.prompting_agent import initial_prompt
from langchain_core.messages import HumanMessage


class StrategySimulator(TradingSimulator):
    """
    TradingSimulator subclass for strategy optimization experiments.

    Differences from TradingSimulator:
    - tool_policy is fixed at initial_prompt (never updated)
    - current_strategy is injected into each trading day's user message
    - strategy_history is tracked and persisted to the log JSON
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_strategy = None
        self.strategy_history = []

    def update_strategy(self, new_strategy: str, date: str = None, reasoning: str = None):
        self.current_strategy = new_strategy
        self.strategy_history.append({
            "date": date,
            "strategy": new_strategy,
            "reasoning": reasoning,
        })
        print(f"[DEBUG] Strategy updated. Preview:")
        print(f"{new_strategy[:200]}...")

    def load_previous_state(self):
        success = super().load_previous_state()
        # Also restore strategy_history from JSON
        try:
            with open(self.log_manager.log_filename, "r") as f:
                log_data = json.load(f)
            if "strategy_history" in log_data and log_data["strategy_history"]:
                self.strategy_history = log_data["strategy_history"]
                self.current_strategy = self.strategy_history[-1]["strategy"]
                print(f"   Restored {len(self.strategy_history)} strategy updates")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        return success

    def save_log_to_json(self):
        # Save base log via parent
        super().save_log_to_json()
        # Append strategy_history to the saved JSON
        if self.strategy_history:
            try:
                with open(self.log_manager.log_filename, "r") as f:
                    log_data = json.load(f)
                log_data["strategy_history"] = self.strategy_history
                with open(self.log_manager.log_filename, "w") as f:
                    json.dump(log_data, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save strategy_history: {e}")

    async def execute_trading(self, date: str):
        """
        Same as TradingSimulator.execute_trading(), but:
        - tool_policy is always fixed at initial_prompt
        - current_strategy is injected into the user message if set
        """
        stock_prices = self.get_stock_prices(date)

        if stock_prices is None:
            print(f"Market closed on {date}. Skipping.")
            return None

        date_obj = datetime.strptime(date, "%Y-%m-%d")
        news_until_date = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")

        prev_equity = self.account.get_total_value()
        self.account.update_prices(stock_prices)

        prev_data = None
        if self.trading_log:
            last_log = self.trading_log[-1]
            prev_allocations = last_log.get("allocations", {})
            prev_date = last_log.get("date", "N/A")
            prev_return = last_log.get("daily_return", 0)
            prev_prices = last_log.get("prices", {})

            asset_returns = {}
            for ticker in prev_allocations.keys():
                if ticker in prev_prices and ticker in stock_prices:
                    asset_returns[ticker] = (
                        (stock_prices[ticker] - prev_prices[ticker]) / prev_prices[ticker]
                    ) * 100

            prev_data = {
                "date": prev_date,
                "allocations": prev_allocations,
                "return": prev_return,
                "asset_returns": asset_returns,
            }

        # Build user message — tool_policy fixed at initial_prompt
        user_message = f"Trading date is {date}.\n"
        user_message += f"IMPORTANT: Price data is available up to {date}, but news data is only available up to {news_until_date}.\n"
        user_message += f"Do NOT use any news from {date} or later."
        user_message += f"\n\n# Tool Usage Policy\n{initial_prompt}"

        # Inject current strategy if available
        if self.current_strategy:
            user_message += f"\n\n# Trading Strategy\n{self.current_strategy}"

        if self.trading_log:
            last_log = self.trading_log[-1]
            prev_allocations = last_log.get("allocations", {})
            prev_date = last_log.get("date", "N/A")
            prev_equity_log = last_log.get("total_equity", 0)
            prev_return = last_log.get("daily_return", 0)

            user_message += f"\n\nPrevious Portfolio State (as of {prev_date}):"
            user_message += f"\n- Total Equity: ${prev_equity_log:,.2f}"
            user_message += f"\n- Daily Return: {prev_return:.2f}%"
            user_message += f"\n- Allocations: {prev_allocations}"

        input_msg = {"messages": [HumanMessage(content=user_message)]}
        response = self.trading_agent.invoke(input=input_msg)
        response_content = response["messages"][-1].content
        print(response_content)

        allocation_data = json.loads(response_content)
        portfolio_allocation = StrategyPortfolioAllocation(**allocation_data)
        traceability = {
            k: {"reasoning": v.reasoning}
            for k, v in portfolio_allocation.traceability.items()
        }
        stock_allocations = portfolio_allocation.allocations

        cur_equity = self.account.apply_allocation(stock_allocations)
        date_df = pd.to_datetime(date)
        self.account.equity.loc[date_df, "equity"] = cur_equity
        daily_return = (cur_equity - prev_equity) / prev_equity if prev_equity > 0 else 0

        print(f"Total Equity: ${cur_equity:,.2f}  Daily Return: {daily_return*100:.2f}%")

        log_entry = {
            "date": date,
            "news_until_date": news_until_date,
            "traceability": traceability,
            "allocations": {k: round(v, 4) for k, v in stock_allocations.items()},
            "positions": {k: round(v, 4) for k, v in self.account.positions.items()},
            "prices": {k: round(v, 2) for k, v in stock_prices.items()},
            "cash": round(self.account.cash_balance, 2),
            "total_equity": round(cur_equity, 2),
            "daily_return": round(daily_return * 100, 4),
        }

        self.trading_log.append(log_entry)
        self.save_log_to_json()

        return {
            "allocations": {k: round(v, 4) for k, v in stock_allocations.items()},
            "traceability": traceability,
            "total_equity": round(cur_equity, 2),
            "daily_return": round(daily_return * 100, 4),
            "prev_data": prev_data,
        }
