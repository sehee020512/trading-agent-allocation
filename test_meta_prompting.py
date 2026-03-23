import json
from pathlib import Path
from agents.prompting_agent import agent_memorizing

MEMORY_PATH = Path("trading_log/2025-04/memory.txt")
LOG_DIRS = [
    Path("trading_log/2025-04"),
]

# 기존 메타프롬프트 로드 (없으면 첫 생성)
if MEMORY_PATH.exists():
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        previous_memory = f.read()
    print("--- Loaded existing memory ---")
else:
    previous_memory = ""
    print("--- No existing memory found. Starting fresh. ---")

# 각 window의 summary를 순차적으로 반영
for log_dir in LOG_DIRS:
    summary_path = log_dir / "summary.txt"
    if not summary_path.exists():
        print(f"Skipping {log_dir.name}: summary.txt not found")
        continue

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = f.read()

    metrics_text = "\n---\n\n# Final Performance Metrics\n\n"
    for run_id in [1, 2, 3]:
        metrics_path = log_dir / f"trading_log_base_run{run_id}.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = data["final_metrics"]
        metrics_text += (
            f"## Run {run_id}\n"
            f"- Cumulative Return: {m['cumulative_return']}%\n"
            f"- Sharpe Ratio: {m['sharpe_ratio']}\n"
            f"- Max Drawdown: {m['max_drawdown']}%\n"
            f"- Win Rate: {m['win_rate']}%\n"
            f"- Volatility: {m['volatility']}%\n\n"
        )

    previous_memory_section = (
        f"# Previous Long-Term Memory\n\n{previous_memory}\n\n---\n\n"
        if previous_memory
        else ""
    )
    message = f"{previous_memory_section}{summary}{metrics_text}"

    print(f"Processing {log_dir.name}...")
    response = agent_memorizing.invoke({"messages": [{"role": "user", "content": message}]})
    previous_memory = response["messages"][-1].content
    print(f"Done: {log_dir.name}")

# 최종 메타프롬프트 저장
MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MEMORY_PATH, "w", encoding="utf-8") as f:
    f.write(previous_memory)

print(f"\n--- Saved to {MEMORY_PATH} ---")
print(previous_memory)
