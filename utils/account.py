import pandas as pd
from typing import Dict

class StockAccount:
    def __init__(self, cash_balance=1000.0):
        self.cash_balance = cash_balance
        self.positions = {}
        self.prices = {}
        self.equity = pd.DataFrame(columns=["equity"])

    def update_prices(self, prices: Dict[str, float]):
        self.prices = prices

    def get_total_value(self):
        stock_value = sum(self.positions.get(ticker, 0) * self.prices.get(ticker, 0)
                         for ticker in self.positions)
        return self.cash_balance + stock_value

    def apply_allocation(self, target_allocations: Dict[str, float]):
        total_value = self.get_total_value()

        print("\n=== [Rebalancing Portfolio] ===")

        all_tickers = set(list(target_allocations.keys()) + list(self.positions.keys()))
        all_tickers.discard("CASH")

        total_cash_needed = 0
        total_cash_released = 0

        for ticker in all_tickers:
            target_ratio = target_allocations.get(ticker, 0.0)
            target_value = total_value * target_ratio
            current_shares = self.positions.get(ticker, 0.0)
            current_price = self.prices.get(ticker, 0.0)
            current_value = current_shares * current_price
            diff_value = target_value - current_value

            if abs(diff_value) < 1:
                continue

            if diff_value > 0:
                total_cash_needed += diff_value
            else:
                total_cash_released += abs(diff_value)

        if total_cash_needed > self.cash_balance + total_cash_released:
            print("⚠️ Not enough cash to complete all trades. Scaling down purchases.")
            scale_factor = (self.cash_balance + total_cash_released) / total_cash_needed
        else:
            scale_factor = 1.0

        for ticker in all_tickers:
            target_ratio = target_allocations.get(ticker, 0.0)
            target_value = total_value * target_ratio
            current_shares = self.positions.get(ticker, 0.0)
            current_price = self.prices.get(ticker, 0.0)

            if current_price == 0:
                continue

            current_value = current_shares * current_price
            diff_value = target_value - current_value

            if abs(diff_value) < 1:
                continue

            if diff_value > 0:
                buy_value = diff_value * scale_factor
                buy_shares = buy_value / current_price
                if self.cash_balance >= buy_value:
                    self.positions[ticker] = self.positions.get(ticker, 0.0) + buy_shares
                    self.cash_balance -= buy_value
                    print(f"BUY {ticker} {buy_shares:.4f} shares @ ${current_price:,.2f}")
            else:
                sell_shares = min(abs(diff_value) / current_price, current_shares)
                self.positions[ticker] = max(0, self.positions.get(ticker, 0.0) - sell_shares)
                sell_value = sell_shares * current_price
                self.cash_balance += sell_value
                print(f"SELL {ticker} {sell_shares:.4f} shares @ ${current_price:,.2f}")

        print(f"\nCash: ${self.cash_balance:,.2f}")
        print(f"Total Value: ${self.get_total_value():,.2f}\n")

        return self.get_total_value()