import asyncio
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = Path(__file__).parent

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pathlib import Path

from utils.account import StockAccount
from utils.strategy_simulator import StrategySimulator
from utils.metric import compute_metrics
from agents.trading_agents import StrategyPortfolioAllocation
from agents.strategy_prompting_agent import StrategyUpdate, strategy_prompting_instruction, initial_strategy

_INSTRUCTIONS_DIR = Path(__file__).parent / "agents" / "instructions"
with open(_INSTRUCTIONS_DIR / "strategy_instruction.txt", "r", encoding="utf-8") as f:
    _strategy_instruction = f.read()

# ── Agent factory ─────────────────────────────────────────────────────────────
def _make_trading_agent():
    from agents.tools import get_price, news_searcher, code_interpreter
    return create_agent(
        model="gpt-5-mini",
        tools=[get_price, news_searcher, code_interpreter],
        system_prompt=_strategy_instruction,
        response_format=StrategyPortfolioAllocation,
    )

def _make_strategy_prompting_agent():
    return create_agent(
        model="gpt-5-mini",
        system_prompt=strategy_prompting_instruction,
        response_format=StrategyUpdate,
    )


# ── Simulation ────────────────────────────────────────────────────────────────
async def run_strategy_simulation(simulator: StrategySimulator, prompting_agent, start_date, end_date):
    """
    Strategy optimization experiment:
    - tool_policy is fixed at initial_prompt (never updated)
    - Trading strategy is optimized daily by the prompting agent
    - No long-term memory used
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Determine starting point
    if simulator.trading_log:
        last_date = datetime.strptime(simulator.trading_log[-1]["date"], "%Y-%m-%d")
        last_date_str = simulator.trading_log[-1]["date"]
        strategy_done_for_last = any(
            s.get("date") == last_date_str for s in simulator.strategy_history
        )
        is_first_day = len(simulator.trading_log) == 1
        if is_first_day or strategy_done_for_last:
            current = last_date + timedelta(days=1)
            print(f"\n🔄 Resuming simulation from {current.strftime('%Y-%m-%d')}")
        else:
            current = last_date
            print(f"\n🔄 Resuming simulation from {current.strftime('%Y-%m-%d')} (strategy retry)")
    else:
        current = start
        print(f"\n🚀 Starting new simulation from {current.strftime('%Y-%m-%d')}")

    print(f"\n📋 Experiment: Strategy Optimization (no memory, fixed tool_policy)")
    print(f"   - tool_policy: fixed at initial_prompt")
    print(f"   - strategy: updated daily by prompting agent\n")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        print(f"\n{'='*100}")
        print(f"===== {date_str} - Running base agent (strategy opt) =====")
        print(f"{'='*100}\n")

        try:
            trading_already_done = simulator.trading_log and simulator.trading_log[-1]["date"] == date_str
            strategy_already_done = any(
                s.get("date") == date_str for s in simulator.strategy_history
            )
            has_prev_data = len(simulator.trading_log) > 1 if trading_already_done else len(simulator.trading_log) > 0

            if trading_already_done:
                if not has_prev_data or strategy_already_done:
                    print(f"⏭️  Already completed {date_str}, skipping to next day")
                    current += timedelta(days=1)
                    continue
                else:
                    print(f"🔄 Trading completed for {date_str}, but strategy update needs retry")
                    last_log = simulator.trading_log[-1]
                    prev_log = simulator.trading_log[-2] if len(simulator.trading_log) > 1 else None
                    prev_data = None
                    if prev_log:
                        prev_prices = prev_log.get("prices", {})
                        current_prices = last_log.get("prices", {})
                        asset_returns = {}
                        for ticker in prev_log.get("allocations", {}).keys():
                            if ticker in prev_prices and ticker in current_prices:
                                asset_returns[ticker] = (
                                    (current_prices[ticker] - prev_prices[ticker]) / prev_prices[ticker]
                                ) * 100
                        prev_data = {
                            "date": prev_log.get("date", "N/A"),
                            "allocations": prev_log.get("allocations", {}),
                            "return": prev_log.get("daily_return", 0),
                            "asset_returns": asset_returns,
                        }
                    agent_output = {
                        "allocations": last_log.get("allocations", {}),
                        "traceability": last_log.get("traceability", {}),
                        "total_equity": last_log.get("total_equity", 0),
                        "daily_return": last_log.get("daily_return", 0),
                        "prev_data": prev_data,
                    }
            else:
                print(f"\n--- Step 1: Trading ---")
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

            # Step 2: Update trading strategy
            if prompting_agent is not None and agent_output.get("prev_data") is not None:
                print(f"\n--- Step 2: Strategy Optimizer ---")
                try:
                    prompting_message = f"Evaluation Date: {date_str}\n\n"
                    prompting_message += "Base Agent Trading Results:\n"
                    prompting_message += "=" * 80 + "\n\n"
                    prompting_message += f"Allocations: {agent_output['allocations']}\n"
                    prompting_message += f"Traceability:\n"
                    for ticker, trace in agent_output["traceability"].items():
                        prompting_message += f"  {ticker}:\n"
                        if "process" in trace:
                            prompting_message += f"    Process: {trace['process']}\n"
                        prompting_message += f"    Reasoning: {trace['reasoning']}\n"
                    prompting_message += f"Total Equity: ${agent_output['total_equity']:,.2f}\n"
                    prompting_message += f"Daily Return: {agent_output['daily_return']:.2f}%\n"

                    if agent_output["prev_data"]:
                        prev = agent_output["prev_data"]
                        prompting_message += f"\nPrevious Performance:\n"
                        prompting_message += f"  Date: {prev.get('date', 'N/A')}\n"
                        prompting_message += f"  Return: {prev.get('return', 0):.2f}%\n"
                        prompting_message += f"  Asset Returns: {prev.get('asset_returns', {})}\n"

                    prompting_message += f"\n\nCurrent Trading Strategy:\n"
                    prompting_message += simulator.current_strategy if simulator.current_strategy else initial_strategy

                    prompting_input = {"messages": [HumanMessage(content=prompting_message)]}
                    prompting_response = prompting_agent.invoke(input=prompting_input)
                    print(f"✅ Strategy optimizer completed\n")

                    response_content = prompting_response["messages"][-1].content
                    parsed_response = json.loads(response_content)
                    reasoning = parsed_response.get("reasoning", "")
                    updated_strategy = parsed_response["strategy"]

                    print(f"💡 Strategy Optimizer Reasoning:\n{reasoning}\n")
                    print(f"📋 Updated Trading Strategy:\n{updated_strategy}\n")

                    simulator.update_strategy(updated_strategy, date=date_str, reasoning=reasoning)
                    simulator.save_log_to_json()

                except Exception as e:
                    print(f"❌ Error in strategy optimizer: {str(e)}")
                    simulator.save_log_to_json()
                    raise Exception(f"Strategy optimization failed on {date_str}: {str(e)}")
            else:
                if agent_output.get("prev_data") is None:
                    print(f"\n⏭️  Skipping strategy optimizer on first trading day")

        except Exception as e:
            print(f"\n❌ Fatal error on {date_str}: {str(e)}")
            raise

        current += timedelta(days=1)

    print("\n" + "=" * 100)
    print("📊 === Final Simulation Summary ===")
    print("=" * 100)

    if simulator.trading_log:
        initial_value = 10000
        final_value = simulator.account.get_total_value()
        profit = final_value - initial_value
        metrics = compute_metrics(simulator.account.equity)

        print(f"\n--- BASE AGENT (strategy opt, no memory) ---")
        print(f"Initial Cash: ${initial_value:,.2f}")
        print(f"Final Value:  ${final_value:,.2f}")
        print(f"Profit:       ${profit:,.2f} ({profit / initial_value * 100:.2f}%)")
        print(f"\nPerformance Metrics:")
        print(f"  Cumulative Return: {metrics['CR']:.2f}%")
        print(f"  Sharpe Ratio:      {metrics['SR']:.4f}")
        print(f"  Max Drawdown:      {metrics['MDD']:.2f}%")
        print(f"  Win Rate:          {metrics['WR']:.2f}%")
        print(f"  Volatility:        {metrics['Vol']:.2f}%")
        print(f"\nStrategy Updates: {len(simulator.strategy_history)}")
        simulator.save_log_to_json()


# ── Single simulation run (프로세스 단위) ────────────────────────────────────
def _run_in_process(run_id: int, initial_cash: float,
                    start_date: str, end_date: str, tickers: list) -> dict:
    agent_name = f"strategy_run{run_id}"
    run_dir = _PROJECT_ROOT / "trading_log" / f"run{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    print(f"\n🚀 [Run {run_id}] 시뮬레이션 시작 (dir: {run_dir})")

    while True:
        try:
            trading_agent = _make_trading_agent()
            prompting_agent = _make_strategy_prompting_agent()
            account = StockAccount(cash_balance=initial_cash)

            simulator = StrategySimulator(
                trading_agent=trading_agent,
                prompting_agent=None,
                account=account,
                tickers=tickers,
                agent_name=agent_name,
                log_dir=".",
            )
            simulator.load_previous_state()

            asyncio.run(
                run_strategy_simulation(simulator, prompting_agent, start_date, end_date)
            )
            break

        except Exception as e:
            print(f"\n🔄 [Run {run_id}] 오류: {e}")
            print(f"💾 로그 저장됨. [Run {run_id}] 재시작...\n")
            continue

    final_value = simulator.account.get_total_value()
    metrics = compute_metrics(simulator.account.equity)

    return {"run_id": run_id, "final_value": final_value, "metrics": metrics}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ###
    NUM_RUNS     = 3
    initial_cash = 10_000
    start_date   = "2025-07-01"
    end_date     = "2025-08-14"
    tickers      = ["AAPL", "MSFT", "NVDA", "JPM", "V", "JNJ", "UNH",
                    "PG", "KO", "XOM", "CAT", "WMT", "META", "TSLA", "AMZN"]
    ###

    print(f"🔄 {NUM_RUNS}개 strategy optimization 시뮬레이션 병렬 실행 중...\n")

    results = []
    with ProcessPoolExecutor(max_workers=NUM_RUNS) as executor:
        futures = {
            executor.submit(
                _run_in_process,
                i + 1, initial_cash, start_date, end_date, tickers,
            ): i + 1
            for i in range(NUM_RUNS)
        }
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"✅ [Run {run_id}] 완료 — Final Value: ${result['final_value']:,.2f}")
            except Exception as e:
                print(f"❌ [Run {run_id}] 실패: {e}")

    results.sort(key=lambda x: x["run_id"])

    _METRIC_LABELS = {
        "CR":  "Cumulative Return",
        "SR":  "Sharpe Ratio",
        "MDD": "Max Drawdown",
        "WR":  "Win Rate",
        "Vol": "Volatility",
    }
    metric_keys = list(_METRIC_LABELS.keys())

    print("\n" + "=" * 100)
    print("📊 === Strategy Optimization Summary ===")
    print("=" * 100)

    for r in results:
        profit = r["final_value"] - initial_cash
        print(f"\n--- Run {r['run_id']} ---")
        print(f"  Final Value:  ${r['final_value']:,.2f}  ({profit / initial_cash * 100:+.2f}%)")
        for k in metric_keys:
            unit = "%" if k in ("CR", "MDD", "WR", "Vol") else ""
            print(f"  {_METRIC_LABELS[k]:<22}: {r['metrics'][k]:.4f}{unit}")

    if results:
        print(f"\n--- Average ({len(results)} runs) ---")
        for k in metric_keys:
            avg = sum(r["metrics"][k] for r in results) / len(results)
            unit = "%" if k in ("CR", "MDD", "WR", "Vol") else ""
            print(f"  {_METRIC_LABELS[k]:<22}: {avg:.4f}{unit}")


if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except Exception as e:
            print(f"\n🔄 오류 발생: {e}")
            print("💾 자동 재시작...\n")
            continue
