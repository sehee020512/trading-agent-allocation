import asyncio
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)

from utils.account import StockAccount
from utils.simulator import TradingSimulator
from utils.metric import compute_metrics
from utils.simulation import run_simulation
from utils.agent_factory import make_trading_agent, make_prompting_agent
from agents.prompting_agent import prompting_instruction_basic

_INSTRUCTIONS_DIR = Path(__file__).parent / "agents" / "instructions"
with open(_INSTRUCTIONS_DIR / "base_instruction.txt", "r", encoding="utf-8") as f:
    _base_instruction = f.read()


# ── Single simulation run (프로세스 단위) ──────────────────────────────────────
def _run_in_process(run_id: int, cfg: dict) -> dict:
    """하나의 시뮬레이션을 독립 프로세스에서 실행.
    프로세스 격리로 메모리·sys.stdout 충돌 없음.
    오류 발생 시 저장된 로그에서 이어서 재시작."""
    agent_name = "base"
    log_dir    = f"{cfg['log_dir']}/run_{run_id}"
    print(f"\n🚀 [Run {run_id}] 시뮬레이션 시작 (log: {log_dir}/trading_log_{agent_name}.json)")

    while True:
        try:
            trading_agent   = make_trading_agent(cfg)
            prompting_agent = make_prompting_agent(cfg)
            account         = StockAccount(cash_balance=cfg["initial_cash"])

            simulator = TradingSimulator(
                trading_agent=trading_agent,
                account=account,
                tickers=cfg["tickers"],
                agent_name=agent_name,
                log_dir=log_dir,
            )
            simulator.load_previous_state()

            asyncio.run(
                run_simulation(
                    simulator, prompting_agent,
                    cfg["start_date"], cfg["end_date"],
                    prompting_interval=cfg["prompting_interval"],
                    memory_file=cfg.get("memory_file"),
                )
            )
            break  # 정상 완료

        except Exception as e:
            print(f"\n🔄 [Run {run_id}] 오류: {e}")
            print(f"💾 로그 저장됨. [Run {run_id}] 재시작...\n")
            continue

    final_value = simulator.account.get_total_value()
    metrics     = compute_metrics(simulator.account.equity)
    return {"run_id": run_id, "final_value": final_value, "metrics": metrics}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ###
    cfg = {
        # Simulation
        "num_runs":           3,
        "prompting_interval": 1,
        "memory_file":        "trading_log/2025-08/memory.txt",  # e.g. "trading_log/meta_prompt.txt" — meta-prompt를 매 prompting마다 주입
        "log_dir":            "trading_log/2025-08_overfitting",
        "initial_cash":       10_000,
        "start_date":         "2025-08-01",
        "end_date":           "2025-08-31",
        "tickers":            ["AAPL", "MSFT", "NVDA", "JPM", "V", "JNJ", "UNH",
                               "PG", "KO", "XOM", "CAT", "WMT", "META", "TSLA", "AMZN"],
        # Trading agent
        "trading_model":      "gpt-5-mini",
        "trading_prompt":     _base_instruction,
        # Prompting agent
        "prompting_model":    "gpt-5-mini",
        "prompting_prompt":   prompting_instruction_basic,
    }
    ###

    # ── Configuration 저장 ─────────────────────────────────────────────────────
    import os
    os.makedirs(cfg["log_dir"], exist_ok=True)
    cfg_path = f"{cfg['log_dir']}/config.txt"
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w", encoding="utf-8") as f:
            for k, v in cfg.items():
                if k in ("trading_prompt", "prompting_prompt"):
                    f.write(f"{k}: (see source)\n")
                else:
                    f.write(f"{k}: {v}\n")
        print(f"📝 Config saved to {cfg_path}\n")

    print(f"🔄 {cfg['num_runs']}개 시뮬레이션 병렬 실행 중...\n")

    results = []
    with ProcessPoolExecutor(max_workers=cfg["num_runs"]) as executor:
        futures = {
            executor.submit(_run_in_process, i + 1, cfg): i + 1
            for i in range(cfg["num_runs"])
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

    # ── 결과 출력 ──────────────────────────────────────────────────────────────
    _METRIC_LABELS = {
        "CR":  "Cumulative Return",
        "SR":  "Sharpe Ratio",
        "MDD": "Max Drawdown",
        "WR":  "Win Rate",
        "Vol": "Volatility",
    }
    metric_keys = list(_METRIC_LABELS.keys())

    print("\n" + "=" * 100)
    print("📊 === Parallel Simulation Summary ===")
    print("=" * 100)

    for r in results:
        profit = r["final_value"] - cfg["initial_cash"]
        print(f"\n--- Run {r['run_id']} ---")
        print(f"  Final Value:  ${r['final_value']:,.2f}  ({profit / cfg['initial_cash'] * 100:+.2f}%)")
        for k in metric_keys:
            unit = "%" if k in ("CR", "MDD", "WR", "Vol") else ""
            print(f"  {_METRIC_LABELS[k]:<22}: {r['metrics'][k]:.4f}{unit}")

    if results:
        n = len(results)
        print(f"\n--- Average ({n} runs) ---")
        for k in metric_keys:
            vals = [r["metrics"][k] for r in results]
            avg  = sum(vals) / n
            var  = sum((v - avg) ** 2 for v in vals) / n
            unit = "%" if k in ("CR", "MDD", "WR", "Vol") else ""
            print(f"  {_METRIC_LABELS[k]:<22}: {avg:.4f}{unit}  (var: {var:.4f})")

        final_vals = [r["final_value"] for r in results]
        avg_fv = sum(final_vals) / n
        var_fv = sum((v - avg_fv) ** 2 for v in final_vals) / n
        print(f"  {'Final Value':<22}: ${avg_fv:,.2f}  (var: {var_fv:,.2f})")

    # ── 정리: JSON 파일 상위 폴더로 이동 후 run 폴더 삭제 ─────────────────────
    print("\n" + "=" * 100)
    _cleanup_runs(cfg["log_dir"], cfg["num_runs"])


# ── Cleanup ────────────────────────────────────────────────────────────────────
def _cleanup_runs(log_dir: str, num_runs: int) -> None:
    """각 run 폴더에서 JSON 파일만 상위 폴더로 이동하고 run 폴더를 삭제."""
    base = Path(log_dir)
    for i in range(1, num_runs + 1):
        run_dir = base / f"run_{i}"
        if not run_dir.exists():
            continue
        for json_file in run_dir.glob("*.json"):
            dest = base / json_file.name
            json_file.rename(dest)
            print(f"📂 이동: {json_file} → {dest}")
        shutil.rmtree(run_dir)
        print(f"🗑️  삭제: {run_dir}")


if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except Exception as e:
            print(f"\n🔄 오류 발생: {e}")
            print("💾 자동 재시작...\n")
            continue
