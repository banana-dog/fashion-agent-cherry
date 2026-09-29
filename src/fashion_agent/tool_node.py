"""The node where the agent reaches outside itself.

Tools are declared, the agent chooses which to run, and whatever comes back goes
into the state so the next prompts can reason about it and the final reply can
cite it. Nothing here invents a reading: an unavailable tool leaves a stated gap
that the reply is expected to mention.
"""

from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from fashion_agent.llm import Context, llm
from fashion_agent.states import FashionState
from fashion_agent.tools import (
    MAX_TOOLS_PER_TURN,
    ToolResult,
    catalogue,
    context_lines,
    hints,
    run_tools,
    sources_ru,
)


def _asking_messages(state: FashionState) -> list:
    """What the model is answering, as a short conversation.

    The request fields go in as the client's own words rather than as a filled-in
    form, so "Москва, до 20000" reads the same here as it does to a person.
    """
    messages = list(state.get("messages") or [])
    hint = _context_hint(state.get("request"))

    if hint:
        return [
            *messages,
            SystemMessage(content=f"Known so far: {hint}"),
        ]

    return messages


class ToolCall(BaseModel):
    tool: str = Field(description="The tool to run, exactly as named in the list")
    place: str | None = Field(
        default=None,
        description="For the weather tool: the city the client named",
    )


class ToolPlanRequest(BaseModel):
    calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tools to run now, at most three, none repeated",
    )


planner = llm.with_structured_output(ToolPlanRequest)

PLANNER_PROMPT = """
You decide which outside tools to run before answering a request.

Rules:
- Run a tool only when the answer would change what you say. A forecast matters
  for something to wear outside; it does not matter for a question about colour.
- Never invent a fact you did not get from a tool. If a tool fails, say that you
  could not check it.
- Prefer nothing over a tool: returning an empty list is a valid answer and is
  usually the right one.
- Use the city the client gave. Do not invent one.
- At most three tools, and never the same one twice.
"""


def _context_hint(request: dict | None) -> str:
    if not request:
        return ""

    parts = []

    for key, label in (
        ("occasion", "повод"),
        ("location", "город"),
        ("budget_max", "бюджет"),
    ):
        value = request.get(key)

        if value:
            parts.append(f"{label}: {value}")

    return "; ".join(parts)


def plan_tools(
    request: dict | None,
    tools: list,
) -> list[dict]:
    try:
        plan: ToolPlanRequest = planner.invoke(
            [
                SystemMessage(
                    content=PLANNER_PROMPT
                    + f"\nAvailable tools:\n{catalogue(tools)}\n"
                ),
                SystemMessage(
                    content=f"Request: {_context_hint(request) or 'nothing yet'}"
                ),
            ]
        )
    except Exception:  # noqa: BLE001 - a turn must survive a planner failure
        return []

    return [call.model_dump() for call in plan.calls]


def check_context(
    state: FashionState,
    runtime: Runtime[Context],
    *,
    tools: list | None = None,
    model=None,
) -> dict:
    """Let the agent reach for what it needs, then leave the findings in state.

    The tools are called natively, so their arguments come from the tools and a
    function with a new parameter needs no change here. The old form-based
    planner stays as a fallback for a model that will not call tools at all.
    """
    available = tools if tools is not None else default_tools()

    if not available:
        return {"tool_results": []}

    try:
        results = _call_tools(available, state, model=model)
    except Exception:  # noqa: BLE001 - a turn must survive a tool-calling failure
        calls = plan_tools(state.get("request"), available)

        if not calls:
            return {"tool_results": []}

        results = run_tools(available, calls)

    if not results:
        return {"tool_results": []}

    return {
        "tool_results": [result.model_dump(mode="json") for result in results],
        "context_lines": context_lines(results),
    }


def _call_tools(
    tools: list,
    state: FashionState,
    *,
    model=None,
) -> list:
    from fashion_agent.tool_calling import ToolCallRunner

    runner = ToolCallRunner(tools, max_calls=MAX_TOOLS_PER_TURN, model=model)

    return runner.run_turn(_asking_messages(state))


def context_lines_from_state(state: FashionState) -> list[str]:
    """What was checked, said plainly, for the client to read."""
    return context_lines(
        [ToolResult.model_validate(item) for item in state.get("tool_results", [])]
    )


def sources_ru_from_state(state: FashionState) -> list[str]:
    return sources_ru(
        [ToolResult.model_validate(item) for item in state.get("tool_results", [])]
    )


def hints_from_state(state: FashionState) -> list[str]:
    return hints(
        [ToolResult.model_validate(item) for item in state.get("tool_results", [])]
    )


_tools: list | None = None


def default_tools() -> list:
    global _tools

    if _tools is None:
        from fashion_agent.product_search.brand_reviews import BrandReviewsTool
        from fashion_agent.tools_weather import WeatherTool

        _tools = [WeatherTool(), BrandReviewsTool()]

    return _tools


def reset_tools() -> None:
    global _tools

    _tools = None
