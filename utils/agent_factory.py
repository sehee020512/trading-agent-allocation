import os
from langchain.agents import create_agent
from langchain_experimental.tools import PythonREPLTool

from agents.trading_agents import PortfolioAllocation
from agents.prompting_agent import PromptUpdate
from agents.tools import get_price, news_searcher


def _is_deepseek(model_str: str) -> bool:
    return isinstance(model_str, str) and model_str.startswith("deepseek:")


def _resolve_model(model_str: str):
    """deepseek:xxx → ChatDeepSeek, 그 외는 문자열 그대로."""
    if _is_deepseek(model_str):
        from langchain_deepseek import ChatDeepSeek
        model_name = model_str.split(":", 1)[1]
        return ChatDeepSeek(
            model=model_name,
            api_key=os.environ["DEEPSEEK_API_KEY"],
        )
    return model_str


def make_trading_agent(cfg):
    kwargs = dict(
        model=_resolve_model(cfg["trading_model"]),
        tools=[get_price, news_searcher, PythonREPLTool()],
        system_prompt=cfg["trading_prompt"],
    )
    if not _is_deepseek(cfg["trading_model"]):
        kwargs["response_format"] = PortfolioAllocation
    return create_agent(**kwargs)


def make_prompting_agent(cfg):
    kwargs = dict(
        model=_resolve_model(cfg["prompting_model"]),
        system_prompt=cfg["prompting_prompt"],
    )
    if not _is_deepseek(cfg["prompting_model"]):
        kwargs["response_format"] = PromptUpdate
    return create_agent(**kwargs)