import json
import os
import re
import pandas as pd
from datetime import datetime


class LogManager:
    """
    Manages trading log persistence and recovery for trading simulators.
    Handles saving/loading trading logs, tool policy history, and account state.
    """

    def __init__(self, agent_name: str, log_dir: str = "trading_log"):
        self.agent_name = agent_name
        self.log_dir = log_dir

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Auto-detect run number and set filenames
        self.load_filename, self.log_filename = self._resolve_filenames()

    def _resolve_filenames(self):
        """
        Find the most recent existing run file (for loading) and
        determine the next run filename (for saving).

        If log_dir contains a run number (e.g. 'run_2' or 'run2'),
        that number is used directly as the file suffix.
        Otherwise, auto-increments from existing files in the directory.

        Returns (load_filename, save_filename).
        """
        base = f"{self.log_dir}/trading_log_{self.agent_name}"

        # Check if log_dir path encodes a run number (e.g. run_2, run2)
        dir_match = re.search(r'run[_\-]?(\d+)', self.log_dir)
        if dir_match:
            run_num = int(dir_match.group(1))
            save_filename = f"{base}_run{run_num}.json"
            # load_filename: same file if it already exists (crash recovery),
            # otherwise None (fresh start)
            load_filename = save_filename if os.path.exists(save_filename) else None
            if load_filename:
                print(f"[LogManager] Resuming existing file: {save_filename}")
            else:
                print(f"[LogManager] No existing file. Saving to: {save_filename}")
            return load_filename, save_filename

        # Fallback: auto-increment based on existing files in the directory
        run = 1
        existing = []
        while os.path.exists(f"{base}_run{run}.json"):
            existing.append(f"{base}_run{run}.json")
            run += 1

        load_filename = existing[-1] if existing else None
        save_filename = f"{base}_run{run}.json"

        if existing:
            print(f"[LogManager] Existing runs found: run1~run{run - 1}. "
                  f"Saving to: {save_filename}")
        else:
            print(f"[LogManager] No existing runs. Saving to: {save_filename}")

        return load_filename, save_filename

    def _merge_tool_policy_history(self, new_history: list) -> list:
        """
        Merge new tool policy history with existing history from file.
        Prevents duplicate entries and preserves chronological order.

        Args:
            new_history: New tool policy history from current session

        Returns:
            Merged tool policy history list
        """
        existing_history = []

        # Load existing history from file
        try:
            with open(self.log_filename, 'r') as f:
                log_data = json.load(f)
                existing_history = log_data.get("tool_policy_history", [])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        if not new_history:
            return existing_history

        if not existing_history:
            return new_history

        # Create a set of existing dates for deduplication
        existing_dates = {entry["date"] for entry in existing_history}

        # Merge: keep existing + add new entries not already present
        merged = existing_history.copy()
        for entry in new_history:
            if entry["date"] not in existing_dates:
                merged.append(entry)
                existing_dates.add(entry["date"])

        # Sort by date
        merged.sort(key=lambda x: x["date"] if x["date"] else "")

        return merged

    def save_log(self, trading_log: list, account, tickers: list,
                 tool_policy_history: list = None):
        """
        Save trading log, account state, and metrics to JSON file.

        Args:
            trading_log: List of daily trading records
            account: Account object with equity history
            tickers: List of ticker symbols
            tool_policy_history: Optional tool policy history (for base agent)
        """
        # Calculate final metrics only
        final_metrics = {}
        if trading_log and len(trading_log) > 0:
            from utils.metric import compute_metrics
            equity_df = account.equity
            metrics = compute_metrics(equity_df)
            final_metrics = {
                "cumulative_return": round(metrics["CR"], 2),
                "sharpe_ratio": round(metrics["SR"], 4),
                "max_drawdown": round(metrics["MDD"], 2),
                "win_rate": round(metrics["WR"], 2),
                "volatility": round(metrics["Vol"], 2)
            }

        log_data = {
            "simulation_info": {
                "agent_name": self.agent_name,
                "start_date": trading_log[0]["date"] if trading_log else None,
                "end_date": trading_log[-1]["date"] if trading_log else None,
                "total_days": len(trading_log),
                "tickers": tickers
            },
            "final_metrics": final_metrics,
            "final_portfolio": {
                "cash": trading_log[-1]["cash"] if trading_log else 0,
                "positions": trading_log[-1]["positions"] if trading_log else {},
                "total_equity": trading_log[-1]["total_equity"] if trading_log else 0
            },
            "daily_logs": trading_log
        }

        # Add tool policy history if present (merge with existing if needed)
        if tool_policy_history is not None:
            merged_history = self._merge_tool_policy_history(tool_policy_history)
            if merged_history:
                log_data["tool_policy_history"] = merged_history

        with open(self.log_filename, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n💾 Trading log saved to: {self.log_filename}")

    def load_log(self, account):
        """
        Load trading log and restore account state from JSON file.

        Args:
            account: Account object to restore state into

        Returns:
            tuple: (trading_log, tool_policy_history, success)
                - trading_log: List of daily trading records
                - tool_policy_history: Tool policy history (None if not base agent)
                - success: Boolean indicating if load was successful
        """
        if self.load_filename is None:
            print(f"\n⚠️ No previous log file found. Starting fresh.")
            return [], None, False

        try:
            with open(self.load_filename, 'r') as f:
                log_data = json.load(f)

            if not log_data.get("daily_logs"):
                print(f"\n⚠️ No trading logs found in {self.load_filename}")
                return [], None, False

            last_day = log_data["daily_logs"][-1]

            # Restore cash balance
            account.cash_balance = last_day["cash"]

            # Restore positions
            account.positions = last_day["positions"].copy()

            # Restore prices
            account.prices = last_day["prices"].copy()

            # Restore equity history
            for daily_log in log_data["daily_logs"]:
                date_df = pd.to_datetime(daily_log["date"])
                account.equity.loc[date_df, "equity"] = daily_log["total_equity"]

            # Get trading log
            trading_log = log_data["daily_logs"].copy()

            # Restore tool policy history if present
            tool_policy_history = None
            if "tool_policy_history" in log_data:
                tool_policy_history = log_data["tool_policy_history"].copy()
                print(f"   Restored {len(tool_policy_history)} tool policy updates")

            print(f"\n✅ Loaded previous state from {self.load_filename}")
            print(f"   Last trading date: {last_day['date']}")
            print(f"   Cash: ${last_day['cash']:,.2f}")
            print(f"   Total equity: ${last_day['total_equity']:,.2f}")
            print(f"   Positions: {len([k for k, v in last_day['positions'].items() if v > 0])} stocks")

            return trading_log, tool_policy_history, True

        except FileNotFoundError:
            print(f"\n⚠️ Previous log file '{self.load_filename}' not found. Starting fresh.")
            return [], None, False
        except Exception as e:
            print(f"\n⚠️ Error loading previous state: {e}")
            return [], None, False
