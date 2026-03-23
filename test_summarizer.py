import json
from pathlib import Path
from agents.prompting_agent import agent_summarizing

LOG_DIR = Path("trading_log/2025-04")

# Load tool_policy_history from each file
histories = []
for run_id in [1, 2, 3]:
    with open(LOG_DIR / f"trading_log_base_run{run_id}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    histories.append(data["tool_policy_history"])

# Build user message
message = json.dumps(
    {
        "run_1_tool_policy_history": histories[0],
        "run_2_tool_policy_history": histories[1],
        "run_3_tool_policy_history": histories[2],
    },
    ensure_ascii=False,
    indent=2,
)

# Invoke agent
response = agent_summarizing.invoke(input={"messages": [{"role": "user", "content": message}]})
result = response["messages"][-1].content

print(result)

output_path = LOG_DIR / "summary.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(result)
print(f"\n--- Saved to {output_path} ---")

