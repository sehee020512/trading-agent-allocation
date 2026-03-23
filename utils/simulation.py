import json
from datetime import datetime, timedelta
from pathlib import Path
from agents.prompting_agent import initial_prompt
from utils.metric import compute_metrics

def _build_accumulated_results(simulator):
    """마지막 prompting 이후의 trading log를 결과 리스트로 변환 (resume 지원)."""
    last_prompt_date = simulator.tool_policy_history[-1]["date"] if simulator.tool_policy_history else None
    results = []
    for i, log in enumerate(simulator.trading_log):
        if last_prompt_date is None or log["date"] > last_prompt_date:
            prev_log = simulator.trading_log[i - 1] if i > 0 else None
            prev_data = None
            if prev_log:
                prev_prices = prev_log.get("prices", {})
                current_prices = log.get("prices", {})
                asset_returns = {
                    ticker: ((current_prices[ticker] - prev_prices[ticker]) / prev_prices[ticker]) * 100
                    for ticker in prev_log.get("allocations", {}).keys()
                    if ticker in prev_prices and ticker in current_prices
                }
                prev_data = {
                    "date": prev_log.get("date", "N/A"),
                    "allocations": prev_log.get("allocations", {}),
                    "return": prev_log.get("daily_return", 0),
                    "asset_returns": asset_returns,
                }
            results.append({
                "date": log["date"],
                "allocations": log.get("allocations", {}),
                "traceability": log.get("traceability", {}),
                "total_equity": log.get("total_equity", 0),
                "daily_return": log.get("daily_return", 0),
                "prev_data": prev_data,
            })
    return results


def _build_prompting_message(accumulated_results, simulator, memory_file=None):
    """N일치 누적 trading 결과로 prompting 메시지 생성.
    memory_file이 있으면 메타프롬프트를 앞에 추가.
    """
    prompting_message = ""
    if memory_file is not None:
        meta_path = Path(memory_file)
        if meta_path.exists():
            meta_content = meta_path.read_text(encoding="utf-8").strip()
            prompting_message += f"# Meta-Prompt (Past Market Regime Insights)\n\n{meta_content}\n\n---\n\n"
            print(f"📂 [Meta-Prompt] Loaded from {meta_path}")
        else:
            print(f"⚠️  [Meta-Prompt] File not found: {meta_path}, proceeding without it")

    start_date = accumulated_results[0]["date"]
    end_date = accumulated_results[-1]["date"]
    prompting_message += f"Evaluation Period: {start_date} to {end_date}\n"
    prompting_message += f"Number of trading days: {len(accumulated_results)}\n\n"
    prompting_message += "Trading Results by Day:\n"
    prompting_message += "=" * 80 + "\n\n"

    for i, result in enumerate(accumulated_results):
        prompting_message += f"Day {i + 1} ({result['date']}):\n"
        prompting_message += f"  Allocations: {result['allocations']}\n"
        prompting_message += f"  Traceability:\n"
        for ticker, trace in result["traceability"].items():
            prompting_message += f"    {ticker}:\n"
            if "process" in trace:
                prompting_message += f"      Process: {trace['process']}\n"
            prompting_message += f"      Reasoning: {trace['reasoning']}\n"
        prompting_message += f"  Total Equity: ${result['total_equity']:,.2f}\n"
        prompting_message += f"  Daily Return: {result['daily_return']:.2f}%\n"
        if result.get("prev_data"):
            prev = result["prev_data"]
            prompting_message += f"  Previous Day ({prev.get('date', 'N/A')}):\n"
            prompting_message += f"    Return: {prev.get('return', 0):.2f}%\n"
            prompting_message += f"    Asset Returns: {prev.get('asset_returns', {})}\n"
        prompting_message += "\n"

    returns = [r["daily_return"] for r in accumulated_results]
    avg_return = sum(returns) / len(returns)
    win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100
    prompting_message += f"Aggregate Stats ({len(accumulated_results)} days):\n"
    prompting_message += f"  Average Daily Return: {avg_return:.2f}%\n"
    prompting_message += f"  Win Rate: {win_rate:.1f}%\n"
    prompting_message += f"  Best Day:  {max(returns):.2f}%\n"
    prompting_message += f"  Worst Day: {min(returns):.2f}%\n\n"

    current_policy = simulator.current_tool_policy or initial_prompt
    prompting_message += "Current Tool Use Policy:\n"
    prompting_message += current_policy

    return prompting_message


async def run_simulation(simulator, prompting_agent, start_date, end_date,
                         prompting_interval=1, memory_file=None):
    """
    Run simulation with continuous prompt updates.
    - Price data: available up to t
    - News data: available up to t-1
    - Decision/Execution/Evaluation: all at t
    - Prompting agent updates tool policy every `prompting_interval` trading days
    - memory_file: optional path to meta-prompt file; if set, its content is prepended to every prompting input
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # 마지막 prompting 이후 누적 결과 복원 (resume 지원)
    accumulated_results = _build_accumulated_results(simulator)

    if simulator.trading_log:
        last_date_str = simulator.trading_log[-1]["date"]
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
        prompting_needed = len(accumulated_results) >= prompting_interval

        if prompting_needed:
            current = last_date
            print(f"\n🔄 Resuming from {last_date_str} (prompting retry, {len(accumulated_results)} days pending)")
        else:
            current = last_date + timedelta(days=1)
            print(f"\n🔄 Resuming simulation from {current.strftime('%Y-%m-%d')}")
            print(f"   Trading days since last prompt: {len(accumulated_results)}/{prompting_interval}")
    else:
        current = start
        print(f"\n🚀 Starting new simulation from {current.strftime('%Y-%m-%d')}")

    print(f"\n📋 Trading Logic:")
    print(f"   - Price data available up to trading date (t)")
    print(f"   - News data available up to previous day (t-1)")
    print(f"   - Decision/Execution/Evaluation all at t")
    print(f"   - Prompting agent updates tool policy every {prompting_interval} trading day(s)")
    print(f"   - Meta-prompt: {'enabled (' + str(memory_file) + ')' if memory_file else 'disabled'}\n")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        print(f"\n{'='*100}")
        print(f"===== {date_str} - Running base agent =====")
        print(f"{'='*100}\n")

        try:
            trading_already_done = simulator.trading_log and simulator.trading_log[-1]["date"] == date_str
            prompting_needed = len(accumulated_results) >= prompting_interval

            if trading_already_done:
                if prompting_needed:
                    print(f"🔄 Trading completed for {date_str}, retrying prompting ({len(accumulated_results)} days accumulated)")
                else:
                    print(f"⏭️  Already completed {date_str}, skipping to next day")
                    current += timedelta(days=1)
                    continue
            else:
                # Step 1: Execute trading
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

                # 시뮬레이션 시작 첫 trading day에는 최적화 건너뜀
                if len(simulator.trading_log) == 1:
                    print(f"\n⏭️  First trading day — skipping optimization (no prior result to evaluate)")
                    current += timedelta(days=1)
                    continue

                # 당일 결과를 누적 리스트에 추가
                accumulated_results = _build_accumulated_results(simulator)

                if len(accumulated_results) < prompting_interval:
                    print(f"\n⏳ Trading days since last prompt: {len(accumulated_results)}/{prompting_interval}")
                    current += timedelta(days=1)
                    continue

            # Step 2: Prompting — 누적 N일치 데이터 사용
            if prompting_agent is not None and accumulated_results:
                print(f"\n--- Step 2: Prompting Agent (every {prompting_interval} day(s)) ---")
                print(f"🔄 Running prompting agent with {len(accumulated_results)} day(s) of results...")

                try:
                    prompting_message = _build_prompting_message(accumulated_results, simulator, memory_file)
                    print(f"📨 [Prompting Agent Input]\n{prompting_message}\n")
                    prompting_input = {"messages": [{"role": "user", "content": prompting_message}]}
                    prompting_response = prompting_agent.invoke(input=prompting_input)

                    # structured_output이 있으면 직접 사용 (일부 모델), 없으면 content에서 JSON 파싱
                    from agents.prompting_agent import PromptUpdate
                    structured = prompting_response.get("structured_output")
                    if isinstance(structured, PromptUpdate):
                        reasoning = structured.reasoning
                        optimized_tool_policy = structured.tool_policy
                    else:
                        response_content = prompting_response["messages"][-1].content
                        parsed_response = json.loads(response_content)
                        reasoning = parsed_response.get("reasoning", "")
                        optimized_tool_policy = parsed_response["tool_policy"]

                    print(f"💡 Prompting Agent Reasoning:\n{reasoning}\n")
                    print(f"📋 Updated Tool Usage Policy:\n{optimized_tool_policy}\n")

                    simulator.update_tool_policy(optimized_tool_policy, date=date_str, reasoning=reasoning)
                    print(f"✅ Tool use policy updated\n")
                    simulator.save_log_to_json()
                    accumulated_results = []

                except Exception as e:
                    print(f"❌ Error in prompting agent: {str(e)}")
                    simulator.save_log_to_json()
                    raise Exception(f"Prompting failed on {date_str}: {str(e)}")

        except Exception as e:
            print(f"\n❌ Fatal error on {date_str}: {str(e)}")
            print(f"💾 Logs saved. You can resume from {date_str} by running the script again.")
            raise

        current += timedelta(days=1)

    # Print final summary
    print("\n" + "=" * 100)
    print("📊 === Final Simulation Summary ===")
    print("=" * 100)

    if simulator.trading_log:
        initial_value = simulator.initial_cash
        final_value = simulator.account.get_total_value()
        profit = final_value - initial_value

        equity_df = simulator.account.equity
        metrics = compute_metrics(equity_df)

        print(f"\n--- BASE AGENT ---")
        print(f"Initial Cash: ${initial_value:,.2f}")
        print(f"Final Value:  ${final_value:,.2f}")
        print(f"Profit:       ${profit:,.2f} ({((final_value - initial_value) / initial_value * 100):.2f}%)")
        print(f"\nPerformance Metrics:")
        print(f"  Cumulative Return: {metrics['CR']:.2f}%")
        print(f"  Sharpe Ratio:      {metrics['SR']:.4f}")
        print(f"  Max Drawdown:      {metrics['MDD']:.2f}%")
        print(f"  Win Rate:          {metrics['WR']:.2f}%")
        print(f"  Volatility:        {metrics['Vol']:.2f}%")

        print(f"\nTool Policy Updates: {len(simulator.tool_policy_history)}")

        simulator.save_log_to_json()
