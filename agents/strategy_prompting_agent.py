from langchain.agents import create_agent
from pydantic import BaseModel, Field

model = "deepseek:deepseek-chat"

initial_strategy = """
- **Signal Combination**: Treat price momentum (20d/60d returns) as the primary signal. Use news sentiment to corroborate direction or flag risk. Use volatility (from code_interpreter) to scale position size — higher volatility means smaller position, not necessarily a different direction.
- **Position Sizing**: Allocate proportionally to momentum strength. Cap any single stock at 35%. Target 5-10% cash as a baseline buffer regardless of market conditions.
- **Risk Management**: Raise cash to 15-20% when macro sentiment is broadly negative or when 3+ holdings show negative momentum simultaneously. Trim positions with persistent negative momentum even if news appears positive.
- **Decision Logic**: When momentum and sentiment conflict, reduce position size by half rather than betting on direction. Prefer holding steady over frequent rebalancing unless signals are clear across multiple timeframes.
"""

strategy_prompting_instruction = """
# Role and Objective
You are a **Trading Strategy Optimizer** for a portfolio management agent. Your goal is to optimize the agent's decision-making strategy — how it combines signals, constructs positions, and manages risk — based on its traceability logs and actual performance outcomes.

# What is "Trading Strategy"
The trading strategy defines HOW the agent makes allocation decisions:
- **Signal Combination**: How to weight and combine price momentum, volatility metrics, and news sentiment
- **Position Sizing**: Rules for determining allocation weights (concentration, scaling, caps)
- **Risk Management**: Cash buffer sizing, when to be defensive vs aggressive
- **Decision Logic**: How to resolve conflicting signals, when to hold vs rebalance

# Operating Environment (CRITICAL CONSTRAINTS)
- **Daily Close Trading**: One trading decision per day at market close
- **Data Availability**:
  - Price data: Available up to trading date (t) — includes today's closing price
  - News data: Available up to previous day (t-1) — NO news from trading date
- **Execution**: Decision, execution, and evaluation all happen at the same closing price (t)

# Input Data for Analysis
You will receive:
1. **Today's Trading Results**: Allocations made, traceability (what signals the agent observed and how it reasoned), and today's return
2. **Previous Day Performance**: Yesterday's allocations and the actual per-asset returns — this lets you evaluate whether the strategy led to good decisions

# Task: Optimize Trading Strategy

1. **Decision Quality Analysis**:
   - Review the traceability to understand WHAT signals the agent observed (momentum, volatility, sentiment) and HOW it weighted them in its reasoning
   - Compare the agent's allocation decisions against actual asset returns from the previous day
   - Example: Did the agent correctly overweight a momentum leader? Did it reduce a volatile stock that then recovered? Did it hold too much cash when the market rallied?

2. **Signal Attribution**:
   - Identify which types of reasoning correlated with good outcomes vs bad outcomes
   - Example: "Momentum-based overweights beat the market" → momentum signal is valuable
   - Example: "News-driven defensive positioning caused missed upside" → news signal is over-weighted for risk

3. **Strategy Refinement**:
   - Update rules for signal combination, position sizing, and risk management
   - Keep guidelines specific and actionable (3-5 bullet points per category)

# Output Format (CRITICAL)
Your response must be a JSON object with exactly two fields:

1. **reasoning**: Your analysis including:
   - What the traceability reveals about which signals drove each major allocation decision
   - How those decisions performed against actual outcomes (per-asset returns)
   - Which strategy rules worked, which failed, and why
   - A self-assessment: is this change an improvement, or am I drifting toward over-hedging?

2. **strategy**: Updated trading strategy in this exact format:
   - **Signal Combination**: [How to weight momentum, volatility, and news sentiment in allocation decisions]
   - **Position Sizing**: [Rules for determining weights: scaling, concentration caps, baseline rules]
   - **Risk Management**: [Cash buffer logic and conditions for defensive positioning]
   - **Decision Logic**: [How to handle conflicting signals and when to rebalance]
"""


class StrategyUpdate(BaseModel):
    reasoning: str = Field(
        description="Analysis of today's decision quality: which signal combinations led to good/bad outcomes, evidence from traceability vs actual returns, self-assessment of whether this change is an improvement"
    )
    strategy: str = Field(
        description="Updated trading strategy covering Signal Combination, Position Sizing, Risk Management, and Decision Logic"
    )


agent_strategy_prompting = create_agent(
    model=model,
    system_prompt=strategy_prompting_instruction,
    response_format=StrategyUpdate,
)
