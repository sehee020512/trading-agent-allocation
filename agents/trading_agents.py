from langchain.agents import create_agent
from pydantic import BaseModel, Field, field_validator
from typing import Dict
from pathlib import Path

from .tools import get_price, news_searcher, code_interpreter


model = "deepseek:deepseek-chat"

### Instructions ####
INSTRUCTIONS_DIR = Path(__file__).parent / "instructions"

with open(INSTRUCTIONS_DIR / "base_instruction.txt", "r", encoding="utf-8") as f:
    base_instruction = f.read()

with open(INSTRUCTIONS_DIR / "baseline_instruction.txt", "r", encoding="utf-8") as f:
    baseline_instruction = f.read()
    
### Structred Output ###
class ToolUseTrace(BaseModel):
    process: str = Field(description="Sequential tool usage: Tool -> Data obtained with date range")
    reasoning: str = Field(description="Logic for this specific asset's weight change")

class StrategyTrace(BaseModel):
    reasoning: str = Field(description="Strategic reason for the weight, referencing signal combination and risk rules")

class StrategyPortfolioAllocation(BaseModel):
    traceability: Dict[str, StrategyTrace] = Field(
        description="Strategic reasoning for each asset's allocation"
    )
    allocations: Dict[str, float] = Field(
        description="A dictionary where keys are assets and values are their respective "
                    "allocation weights. IMPORTANT: The sum of all values MUST be exactly 1.0. "
                    "Example: {'AAPL': 0.25, 'MSFT': 0.20, 'NVDA': 0.15, 'CASH': 0.40}"
    )

    @field_validator('allocations', mode='after')
    @classmethod
    def ensure_normalization(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if total <= 0:
            raise ValueError("Total allocation sum must be greater than 0")

        if abs(total - 1.0) > 0.01:
            return {k: round(float(val) / total, 4) for k, val in v.items()}
        return v

class PortfolioAllocation(BaseModel):
    analysis: str = Field(default="", description="Brief analysis summary before the allocation")
    traceability: Dict[str, ToolUseTrace] = Field(
        description="Detailed trace of tool usage and logic for each asset"
    )
    allocations: Dict[str, float] = Field(
        description="A dictionary where keys are assets and values are their respective "
                    "allocation weights. IMPORTANT: The sum of all values MUST be exactly 1.0. "
                    "Example: {'AAPL': 0.25, 'MSFT': 0.20, 'NVDA': 0.15, 'CASH': 0.40}"
    )

    @field_validator('allocations', mode='after')
    @classmethod
    def ensure_normalization(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if total <= 0:
            raise ValueError("Total allocation sum must be greater than 0")

        if abs(total - 1.0) > 0.01:
            return {k: round(float(val) / total, 4) for k, val in v.items()}
        return v

### Create Agent ###
agent_base = create_agent(
    model = model,
    tools = [get_price,
             news_searcher,
             code_interpreter,
             ],
    system_prompt = base_instruction,
    # response_format = PortfolioAllocation
)

agent_baseline = create_agent(
    model = model,
    tools = [get_price,
             news_searcher,
             code_interpreter,
             ],
    system_prompt = baseline_instruction,
    response_format = PortfolioAllocation
)