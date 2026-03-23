import asyncio
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)

from utils.account import StockAccount
from utils.simulator import TradingSimulator
from utils.simulation import run_simulation
from utils.agent_factory import make_trading_agent, make_prompting_agent
from agents.prompting_agent import prompting_instruction_basic

_INSTRUCTIONS_DIR = Path(__file__).parent / "agents" / "instructions"
with open(_INSTRUCTIONS_DIR / "base_instruction.txt", "r", encoding="utf-8") as f:
    _base_instruction = f.read()


# ── Main ───────────────────────────────────────────────────────────────────────
async def main():

    ###
    cfg = {
        # Simulation
        "agent_name":         "deepseek",
        "log_dir":            "trading_log/deepseek",
        "initial_cash":       10_000,
        "start_date":         "2025-01-01",
        "end_date":           "2025-01-31",
        "tickers":            ["AAPL", "MSFT", "NVDA", "JPM", "V", "JNJ", "UNH",
                               "PG", "KO", "XOM", "CAT", "WMT", "META", "TSLA", "AMZN"],
        "prompting_interval": 1,
        "memory_file":        None,  # e.g. "trading_log/meta_prompt.txt" — meta-prompt를 매 prompting마다 주입
        # Trading agent
        "trading_model":      "deepseek:deepseek-chat",
        "trading_prompt":     _base_instruction,
        # Prompting agent
        "prompting_model":    "deepseek:deepseek-chat",
        "prompting_prompt":   prompting_instruction_basic,
    }
    ###

    trading_agent   = make_trading_agent(cfg)
    prompting_agent = make_prompting_agent(cfg)
    account         = StockAccount(cash_balance=cfg["initial_cash"])

    simulator = TradingSimulator(
        trading_agent=trading_agent,
        account=account,
        tickers=cfg["tickers"],
        agent_name=cfg["agent_name"],
        log_dir=cfg["log_dir"],
    )
    simulator.load_previous_state()

    await run_simulation(
        simulator, prompting_agent,
        cfg["start_date"], cfg["end_date"],
        prompting_interval=cfg["prompting_interval"],
        memory_file=cfg["memory_file"],
    )


if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())
            break
        except Exception as e:
            print(f"\n🔄 Error occurred: {str(e)}")
            print(f"💾 Logs saved. Automatically restarting...\n")
            continue
