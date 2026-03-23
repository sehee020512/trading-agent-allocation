import json
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from agents.trading_agents import PortfolioAllocation
from utils.log_manager import LogManager

class TradingSimulator:
    def __init__(self, trading_agent, account, tickers, agent_name="default", log_dir="trading_log"):
        self.trading_agent = trading_agent
        self.account = account
        self.initial_cash = account.cash_balance
        self.tickers = tickers
        self.trading_log = []
        self.agent_name = agent_name

        # Initialize log manager
        self.log_manager = LogManager(agent_name, log_dir=log_dir)

        # Store original system prompt (strategy part - unchanging)
        self.original_system_prompt = trading_agent.config.get("configurable", {}).get("system_prompt", "")

        # Current tool use policy (will be updated by prompting agent)
        self.current_tool_policy = None

        # Tool policy history (only for base agent)
        self.tool_policy_history = []


    def load_previous_state(self):
        """Load the last portfolio state from a previous trading log"""
        # Use LogManager to load state
        trading_log, tool_policy_history, success = self.log_manager.load_log(self.account)

        if success:
            self.trading_log = trading_log

            # Restore tool policy history for base agent
            if tool_policy_history:
                self.tool_policy_history = tool_policy_history
                # Restore the last tool policy
                self.current_tool_policy = self.tool_policy_history[-1]["policy"]

        return success

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

    def update_tool_policy(self, new_tool_policy: str, date: str = None, reasoning: str = None):
        """
        Update the Tool Usage Policy that will be included in user messages.
        """
        self.current_tool_policy = new_tool_policy

        self.tool_policy_history.append({
            "date": date,
            "policy": new_tool_policy,
            "reasoning": reasoning
        })

        print(f"[DEBUG] Tool policy updated. Preview:")
        print(f"{new_tool_policy[:200]}...")

    async def execute_trading(self, date: str):
        """
        Execute trading for the given date.
        - Price data: available up to date (t)
        - News data: available up to date - 1 (t-1)
        - Decision/Execution/Evaluation: all at date's closing price

        Args:
            date: The trading date (t)

        Returns:
            dict with execution results, or None if market closed
        """
        stock_prices = self.get_stock_prices(date)

        if stock_prices is None:
            print(f"Market closed on {date}. Skipping.")
            return None

        # Calculate news_until_date (t-1) - news is restricted to previous day
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        news_until_date = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")

        # Get previous equity BEFORE updating prices
        prev_equity = self.account.get_total_value()

        self.account.update_prices(stock_prices)

        # Prepare previous performance data if available
        prev_data = None
        if self.trading_log:
            last_log = self.trading_log[-1]
            prev_allocations = last_log.get("allocations", {})
            prev_date = last_log.get("date", "N/A")
            prev_return = last_log.get("daily_return", 0)
            prev_prices = last_log.get("prices", {})

            # Calculate actual performance of each asset
            asset_returns = {}
            for ticker in prev_allocations.keys():
                if ticker in prev_prices and ticker in stock_prices:
                    asset_returns[ticker] = ((stock_prices[ticker] - prev_prices[ticker]) / prev_prices[ticker]) * 100

            prev_data = {
                "date": prev_date,
                "allocations": prev_allocations,
                "return": prev_return,
                "asset_returns": asset_returns
            }

        # Build user message for trading agent
        user_message = f"Trading date is {date}.\n"
        user_message += f"IMPORTANT: Price data is available up to {date}, but news data is only available up to {news_until_date}.\n"
        user_message += f"Do NOT use any news from {date} or later."

        if self.current_tool_policy:
            user_message += f"\n\n# Tool Usage Policy\n{self.current_tool_policy}"
        else:
            from agents.prompting_agent import initial_prompt
            user_message += f"\n\n# Tool Usage Policy\n{initial_prompt}"

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

        # Execute trading agent (astream: 툴 하나 완료될 때마다 실시간 출력)
        print(f"📨 [Trading Agent Input]\n{user_message}\n")
        input_msg = {"messages": [{"role": "user", "content": user_message}]}
        # stream_mode="values": 노드 완료마다 누적 state 전달 → 마지막 청크 = 최종 state
        response = None
        prev_msg_count = 0
        async for state in self.trading_agent.astream(input_msg, stream_mode="values"):
            response = state
            new_msgs = state.get("messages", [])[prev_msg_count:]
            prev_msg_count = len(state.get("messages", []))
            for msg in new_msgs:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"🔧 Tool call: {tc['name']}  args={tc['args']}", flush=True)
                elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    preview = str(msg.content)[:300]
                    print(f"📊 Tool result: {preview}{'...' if len(str(msg.content)) > 300 else ''}", flush=True)
                elif msg.content:
                    print(f"💬 {msg.content}", flush=True)

        # 모든 메시지 raw 출력
        print("\n[DEBUG] === All messages ===")
        for i, msg in enumerate(response.get("messages", [])):
            tc = getattr(msg, "tool_calls", None)
            print(f"  [{i}] {type(msg).__name__} | content={repr(msg.content)[:100]} | tool_calls={tc}")
        print("[DEBUG] === End messages ===\n")

        # structured_output 키 → tool call args → content JSON 순으로 파싱
        structured = response.get("structured_output")
        if isinstance(structured, PortfolioAllocation):
            portfolio_allocation = structured
        else:
            portfolio_allocation = None
            for msg in reversed(response.get("messages", [])):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc["name"] == PortfolioAllocation.__name__:
                            portfolio_allocation = PortfolioAllocation(**tc["args"])
                            break
                if portfolio_allocation:
                    break
            if portfolio_allocation is None:
                portfolio_allocation = PortfolioAllocation(**json.loads(response["messages"][-1].content))
        print(f"🤖 [Trading Agent Output]\n{portfolio_allocation.model_dump_json(indent=2)}\n")
        traceability = {k: {"process": v.process, "reasoning": v.reasoning} for k, v in portfolio_allocation.traceability.items()}

        # Tool call 로그 추출
        tool_call_log = []
        for msg in response.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_call_log.append({"tool": tc["name"], "args": tc["args"]})
            elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
                if tool_call_log and "result" not in tool_call_log[-1]:
                    tool_call_log[-1]["result"] = msg.content

        stock_allocations = portfolio_allocation.allocations

        # Apply allocation and calculate daily return
        cur_equity = self.account.apply_allocation(stock_allocations)
        date_df = pd.to_datetime(date)
        self.account.equity.loc[date_df, "equity"] = cur_equity

        daily_return = (cur_equity - prev_equity) / prev_equity if prev_equity > 0 else 0

        # Print current equity
        print(f"Total Equity: ${cur_equity:,.2f}  Daily Return: {daily_return*100:.2f}%")

        # Save to log
        log_entry = {
            "date": date,
            "news_until_date": news_until_date,
            "tool_call_log": tool_call_log,
            "traceability": traceability,
            "allocations": {k: round(v, 4) for k, v in stock_allocations.items()},
            "positions": {k: round(v, 4) for k, v in self.account.positions.items()},
            "prices": {k: round(v, 2) for k, v in stock_prices.items()},
            "cash": round(self.account.cash_balance, 2),
            "total_equity": round(cur_equity, 2),
            "daily_return": round(daily_return * 100, 4)
        }

        self.trading_log.append(log_entry)
        self.save_log_to_json()

        # Return output for prompting agent
        return {
            "allocations": {k: round(v, 4) for k, v in stock_allocations.items()},
            "traceability": traceability,
            "total_equity": round(cur_equity, 2),
            "daily_return": round(daily_return * 100, 4),
            "prev_data": prev_data
        }

    def save_log_to_json(self):
        """Save trading log using LogManager"""
        self.log_manager.save_log(
            trading_log=self.trading_log,
            account=self.account,
            tickers=self.tickers,
            tool_policy_history=self.tool_policy_history if self.tool_policy_history else None
        )