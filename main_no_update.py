import asyncio
import pandas as pd
import os
from datetime import datetime, timedelta
from utils.account import StockAccount
from utils.simulator import TradingSimulator
from agents.trading_agents import agent_baseline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


async def run_no_update_simulation(simulator, start_date, end_date):
    """
    Run simulation for no_update_base agent.
    - Price data: available up to t
    - News data: available up to t-1
    - Decision/Execution/Evaluation: all at t

    This agent receives optimized tool policies from the prompting agent,
    but does NOT apply them - keeping the initial policy throughout.
    This serves as a control group to measure the impact of policy updates.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Determine starting point
    if simulator.trading_log:
        last_date = datetime.strptime(simulator.trading_log[-1]["date"], "%Y-%m-%d")
        current = last_date + timedelta(days=1)
        print(f"\n🔄 Resuming simulation from {current.strftime('%Y-%m-%d')}")
        print(f"   Last completed date: {last_date.strftime('%Y-%m-%d')}")
    else:
        current = start
        print(f"\n🚀 Starting new simulation from {current.strftime('%Y-%m-%d')}")

    print(f"\n📋 Trading Logic:")
    print(f"   - Price data available up to trading date (t)")
    print(f"   - News data available up to previous day (t-1)")
    print(f"   - Decision/Execution/Evaluation all at t\n")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        print(f"\n{'='*100}")
        print(f"===== {date_str} - Running no_update_base agent =====")
        print(f"{'='*100}\n")

        try:
            # Check if already completed this date
            if simulator.trading_log and simulator.trading_log[-1]["date"] == date_str:
                print(f"⏭️  Already completed {date_str}, skipping to next day")
                current += timedelta(days=1)
                continue

            # Execute trading
            try:
                agent_output = await simulator.execute_trading(date_str)

                if agent_output is None:
                    print(f"📅 Market closed on {date_str}. Skipping to next trading day.")
                    current += timedelta(days=1)
                    continue

                print(f"✅ Trading completed")
                print(f"   Allocations: {agent_output['allocations']}")
                print(f"   Total Equity: ${agent_output['total_equity']:,.2f}")
                print(f"   Daily Return: {agent_output['daily_return']:.2f}%")

            except Exception as e:
                print(f"❌ Error in trading: {str(e)}")
                simulator.save_log_to_json()
                raise Exception(f"Trading failed on {date_str}: {str(e)}")

        except Exception as e:
            print(f"\n❌ Fatal error on {date_str}: {str(e)}")
            print(f"💾 Logs saved. You can resume from {date_str} by running the script again.")
            raise

        current += timedelta(days=1)

    # Print final summary
    print("\n" + "="*100)
    print("📊 === Final Simulation Summary ===")
    print("="*100)

    if simulator.trading_log:
        initial_value = 10000
        final_value = simulator.account.get_total_value()
        profit = final_value - initial_value

        # Calculate final metrics
        from utils.metric import compute_metrics
        equity_df = simulator.account.equity
        metrics = compute_metrics(equity_df)

        print(f"\n--- NO_UPDATE_BASE AGENT ---")
        print(f"Initial Cash: ${initial_value:,.2f}")
        print(f"Final Value:  ${final_value:,.2f}")
        print(f"Profit:       ${profit:,.2f} ({((final_value - initial_value) / initial_value * 100):.2f}%)")
        print(f"\nPerformance Metrics:")
        print(f"  Cumulative Return: {metrics['CR']:.2f}%")
        print(f"  Sharpe Ratio:      {metrics['SR']:.4f}")
        print(f"  Max Drawdown:      {metrics['MDD']:.2f}%")
        print(f"  Win Rate:          {metrics['WR']:.2f}%")
        print(f"  Volatility:        {metrics['Vol']:.2f}%")

        print(f"\nPolicy Updates Received (but not applied): {len(simulator.tool_policy_history)}")

        simulator.save_log_to_json()


async def main():

    ###
    initial_cash = 10000
    start_date = "2025-02-01"
    end_date = "2025-02-28"
    tickers = ["AAPL", "MSFT", "NVDA", "JPM", "V", "JNJ", "UNH", "PG", "KO", "XOM", "CAT", "WMT", "META", "TSLA", "AMZN"]
    ###

    # Initialize no_update_base agent
    account = StockAccount(cash_balance=initial_cash)
    simulator = TradingSimulator(
        trading_agent=agent_baseline,
        account=account,
        tickers=tickers,
        agent_name="no_update_base"
    )
    simulator.load_previous_state()

    # Run simulation
    await run_no_update_simulation(simulator, start_date, end_date)


if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())
            break  # 정상 완료시 종료
        except Exception as e:
            print(f"\n🔄 오류 발생: {str(e)}")
            print(f"💾 로그가 저장되었습니다. 자동으로 재시작합니다...\n")
            continue
