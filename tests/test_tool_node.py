"""The node where the agent reaches outside itself."""

from typing import ClassVar

from fashion_agent.tool_node import (
    ToolCall,
    ToolPlanRequest,
    check_context,
    context_lines_from_state,
    default_tools,
    hints_from_state,
    plan_tools,
    reset_tools,
    sources_ru_from_state,
)
from fashion_agent.tools import ToolResult, ToolUnavailable


class Stub:
    name = "stub"
    description = "a stub"
    parameters: ClassVar[list[str]] = ["place"]

    def __init__(self, result=None, error=None, available=True):
        self.result = result or ToolResult(tool=self.name, lines=["ок"])
        self.error = error
        self.available_flag = available
        self.calls: list[dict] = []

    def available(self):
        return self.available_flag

    def run(self, **params):
        self.calls.append(params)

        if self.error:
            raise self.error

        return self.result


def with_plan(monkeypatch, *calls) -> None:
    plan = ToolPlanRequest(calls=[ToolCall(**call) for call in calls])
    monkeypatch.setattr(
        "fashion_agent.tool_node.planner",
        type("Planner", (), {"invoke": staticmethod(lambda _m: plan)})(),
    )


def test_no_tools_means_nothing_runs():
    assert check_context({"request": {}}, None, tools=[]) == {"tool_results": []}


def test_the_agent_may_choose_no_tool(monkeypatch):
    with_plan(monkeypatch)
    tool = Stub()

    result = check_context({"request": {}}, None, tools=[tool])

    assert result == {"tool_results": []}
    assert tool.calls == []


def test_a_chosen_tool_runs_with_the_given_parameters(monkeypatch):
    with_plan(monkeypatch, {"tool": "stub", "place": "Москва"})
    tool = Stub()

    result = check_context(
        {"request": {"location": "Москва"}},
        None,
        tools=[tool],
    )

    assert tool.calls == [{"place": "Москва"}]
    assert result["tool_results"][0]["tool"] == "stub"
    assert result["context_lines"]


def test_a_finding_reaches_the_next_prompts(monkeypatch):
    with_plan(monkeypatch, {"tool": "stub", "place": "Москва"})
    tool = Stub(
        result=ToolResult(
            tool="stub",
            lines=["Москва: дождь, 4 °C"],
            hints=["обувь, которая не боится воды"],
        )
    )

    result = check_context({"request": {}}, None, tools=[tool])
    state = {"request": {}, **result}

    assert "дождь, 4 °C" in " ".join(context_lines_from_state(state))
    assert hints_from_state(state) == ["обувь, которая не боится воды"]


def test_a_failed_tool_is_recorded_not_swallowed(monkeypatch):
    with_plan(monkeypatch, {"tool": "stub"})
    tool = Stub(error=ToolUnavailable("not reachable"))

    result = check_context({"request": {}}, None, tools=[tool])

    assert result["tool_results"][0]["ok"] is False
    assert "not reachable" in result["tool_results"][0]["error"]
    assert result["context_lines"] == []


def test_the_client_is_told_which_source_was_used(monkeypatch):
    with_plan(monkeypatch, {"tool": "stub"})
    tool = Stub(
        result=ToolResult(
            tool="stub",
            lines=["Москва: ясно"],
            source_name="MET Norway",
            source_url="https://api.met.no/doc/Weatherapi",
        )
    )

    state = {"request": {}, **check_context({"request": {}}, None, tools=[tool])}

    assert "MET Norway" in sources_ru_from_state(state)[0]


def test_the_client_is_told_when_a_check_failed(monkeypatch):
    with_plan(monkeypatch, {"tool": "stub"})
    tool = Stub(error=ToolUnavailable("not reachable"))

    state = {"request": {}, **check_context({"request": {}}, None, tools=[tool])}

    assert "проверить не удалось" in sources_ru_from_state(state)[0]


def test_a_planner_failure_leaves_the_turn_alone(monkeypatch):
    def boom(_messages):
        raise RuntimeError("model down")

    monkeypatch.setattr(
        "fashion_agent.tool_node.planner",
        type("Planner", (), {"invoke": staticmethod(boom)})(),
    )

    assert check_context({"request": {}}, None, tools=[Stub()]) == {"tool_results": []}


def test_the_request_travels_to_the_planner(monkeypatch):
    seen: list[str] = []

    def capture(messages):
        seen.extend(message.content for message in messages)

        return ToolPlanRequest(calls=[])

    monkeypatch.setattr(
        "fashion_agent.tool_node.planner",
        type("Planner", (), {"invoke": staticmethod(capture)})(),
    )

    plan_tools({"occasion": "на концерт", "location": "Казань"}, [Stub()])

    joined = " ".join(seen)

    assert "на концерт" in joined
    assert "Казань" in joined


def test_the_catalogue_is_offered_to_the_planner(monkeypatch):
    seen: list[str] = []

    def capture(messages):
        seen.extend(message.content for message in messages)

        return ToolPlanRequest(calls=[])

    monkeypatch.setattr(
        "fashion_agent.tool_node.planner",
        type("Planner", (), {"invoke": staticmethod(capture)})(),
    )

    plan_tools({}, [Stub()])

    assert "stub(place)" in " ".join(seen)


def test_the_prompt_forbids_inventing_and_bounding(caplog):
    from fashion_agent.tool_node import PLANNER_PROMPT

    assert "Never invent a fact" in PLANNER_PROMPT
    assert "empty list" in PLANNER_PROMPT
    assert "at most three" in PLANNER_PROMPT.lower()
    assert "Do not invent one" in PLANNER_PROMPT


def test_default_tools_include_weather():
    reset_tools()
    tools = default_tools()

    assert [tool.name for tool in tools] == ["weather"]


def test_an_empty_request_still_reaches_the_planner(monkeypatch):
    seen: list[str] = []

    def capture(messages):
        seen.extend(message.content for message in messages)

        return ToolPlanRequest(calls=[])

    monkeypatch.setattr(
        "fashion_agent.tool_node.planner",
        type("Planner", (), {"invoke": staticmethod(capture)})(),
    )

    plan_tools(None, [Stub()])

    assert "nothing yet" in " ".join(seen)


def test_a_missing_result_key_is_tolerated():
    assert context_lines_from_state({}) == []
    assert sources_ru_from_state({}) == []
    assert hints_from_state({}) == []


def test_the_search_plan_is_told_what_was_checked(monkeypatch):
    state = {
        "request": {"occasion": "на работу", "location": "Москва"},
        "context_lines": ["weather: Москва: дождь, 4 °C"],
    }

    from fashion_agent.product_search.product_search import create_search_plan

    captured: list[str] = []

    class Recorder:
        def __init__(self, plan):
            self.plan = plan

        def invoke(self, messages):
            captured.extend(
                message.content for message in messages
            )

            return self.plan

    from src.fashion_agent.product_search.models import ProductSearch, SearchPlan

    search = ProductSearch(
        category="outerwear",
        query="плащ",
        fallback_query="плащ",
    )
    monkeypatch.setattr(
        "fashion_agent.product_search.product_search.search_plan_extractor",
        Recorder(SearchPlan(searches=[search])),
    )

    create_search_plan(
        {
            **state,
            "style_preferences": [],
            "client_profile": {},
            "wardrobe_items": [],
            "resolved_style": None,
            "retrieved_outfit_formulas": [],
            "retrieved_trends": [],
        },
        type("Runtime", (), {"context": type("C", (), {"locale": "ru-RU", "currency": "RUB"})()})(),
    )

    prompt = " ".join(captured)

    assert "Москва: дождь, 4 °C" in prompt
    assert "coat is not optional" in prompt
