"""The agent calling tools itself.

The thing being protected is that the model, not a hand-written list, decides
what to call and with which arguments. The rest is about the loop ending: a
bounded budget, one answer per call, and a failure that does not invite a retry
storm.
"""

from typing import ClassVar

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from fashion_agent.tool_calling import (
    MAX_ROUNDS,
    ToolCallRunner,
    describe,
    tools_from_response,
)
from fashion_agent.tools import (
    ToolResult,
    structured_tools,
)


class Args(BaseModel):
    subject: str = Field(description="Что именно")


class Probe:
    """A tool that records what it was asked for."""

    name = "probe"
    description = "Спрашивает что-нибудь и запоминает"
    parameters: ClassVar[list] = ["subject"]
    calls: ClassVar[list] = []

    def args_model(self):
        return Args

    def available(self) -> bool:
        return True

    def run(self, subject: str | None = None, **_: object) -> ToolResult:
        Probe.calls.append(subject)

        return ToolResult(
            tool=self.name,
            lines=[f"по теме «{subject}» есть ответ"],
            source_name="проба",
        )


class Failing(Probe):
    name = "failing"
    calls: ClassVar[list] = []

    def run(self, subject: str | None = None, **_: object) -> ToolResult:
        Failing.calls.append(subject)

        raise RuntimeError("источник упал")


class Unavailable(Probe):
    name = "unavailable"
    calls: ClassVar[list] = []

    def available(self) -> bool:
        return False

    def run(self, subject: str | None = None, **_: object) -> ToolResult:
        Unavailable.calls.append(subject)

        return ToolResult(tool=self.name, lines=["не должен вызываться"])


def call(name: str, call_id: str, **arguments):
    return {"name": name, "args": arguments, "id": call_id, "type": "tool_call"}


class FakeModel:
    """A model that answers with a scripted list of tool calls."""

    def __init__(self, script: list[list[dict]]):
        self.script = list(script)
        self.seen_tools: list[list[str]] = []
        self.rounds = 0

    def bind_tools(self, tools):
        self.seen_tools.append([tool.name for tool in tools])

        return self

    def invoke(self, messages):
        self.rounds += 1
        wanted = self.script.pop(0) if self.script else []

        return AIMessage(
            content="",
            tool_calls=[call(entry["name"], entry.get("id", f"c{self.rounds}"), **entry.get("args", {})) for entry in wanted]
            if wanted
            else [],
        )


class TestSchemas:
    def test_a_tool_describes_its_own_arguments(self):
        [tool] = structured_tools([Probe()])

        assert tool.args_schema.model_json_schema()["required"] == ["subject"]

    def test_a_tool_with_no_arguments_is_offered(self):
        class Bare:
            name = "bare"
            description = "Ничего не спрашивает"
            parameters: ClassVar[list] = []

            def args_model(self):
                from fashion_agent.tools import NoArguments

                return NoArguments

            def available(self) -> bool:
                return True

            def run(self, **_: object) -> ToolResult:
                return ToolResult(tool=self.name, lines=["готово"])

        [tool] = structured_tools([Bare()])

        assert tool.args_schema.model_json_schema().get("properties", {}) == {}

    def test_a_tool_that_cannot_answer_is_not_offered(self):
        """Offering it wastes a turn on a refusal the model could predict."""
        assert structured_tools([Unavailable()]) == []

    def test_the_description_says_when_a_tool_is_down(self):
        listed = describe([Unavailable()])

        assert "сейчас недоступен" in listed

    def test_the_description_names_the_arguments(self):
        assert "subject" in describe([Probe()])


class TestOneRound:
    def test_the_model_asks_and_the_tool_runs(self):
        Probe.calls = []
        model = FakeModel([[{"name": "probe", "args": {"subject": "погода"}}]])

        results = ToolCallRunner([Probe()], max_calls=3, model=model).run_turn([])

        assert Probe.calls == ["погода"]
        assert results[0].ok is True

    def test_the_model_is_offered_the_tools(self):
        model = FakeModel([[]])

        ToolCallRunner([Probe(), Unavailable()], max_calls=3, model=model).run_turn([])

        assert model.seen_tools[0] == ["probe"]

    def test_nothing_asked_means_nothing_run(self):
        Probe.calls = []
        model = FakeModel([[]])

        results = ToolCallRunner([Probe()], max_calls=3, model=model).run_turn([])

        assert results == []
        assert Probe.calls == []

    def test_a_failed_tool_is_reported_as_a_failure(self):
        """An empty answer would let the model guess, which is the point of the tool."""
        Failing.calls = []
        model = FakeModel([[{"name": "failing", "args": {"subject": "x"}}]])

        [result] = ToolCallRunner([Failing()], max_calls=3, model=model).run_turn([])

        assert result.ok is False
        # The class name only: this string is shown to a client, and a provider's
        # message can carry an account name or an internal host.
        assert result.error == "RuntimeError"

    def test_a_failed_tool_never_answers_with_a_blank_result(self):
        """An empty answer would let the model fill the gap from its memory."""
        from fashion_agent.tool_calling import _answer

        said = _answer(
            ToolResult(tool="failing", ok=False, error="RuntimeError")
        )

        assert said != ""
        assert "недоступен" in said


class TestBudget:
    def test_the_same_call_is_not_made_twice(self):
        Probe.calls = []
        model = FakeModel(
            [
                [{"name": "probe", "args": {"subject": "дождь"}}],
                [{"name": "probe", "args": {"subject": "дождь"}}],
            ]
        )

        results = ToolCallRunner([Probe()], max_calls=5, model=model).run_turn([])

        assert Probe.calls == ["дождь"]
        assert len(results) == 1

    def test_two_calls_to_one_tool_keep_the_first(self):
        Probe.calls = []
        model = FakeModel(
            [
                [
                    {"name": "probe", "args": {"subject": "раз"}},
                    {"name": "probe", "args": {"subject": "два"}},
                ]
            ]
        )

        results = ToolCallRunner([Probe()], max_calls=5, model=model).run_turn([])

        assert Probe.calls == ["раз"]
        assert len(results) == 1

    def test_the_turn_stops_at_its_budget(self):
        Probe.calls = []
        model = FakeModel(
            [
                [
                    {"name": "probe", "args": {"subject": "a"}},
                    {"name": "probe", "args": {"subject": "b"}},
                    {"name": "probe", "args": {"subject": "c"}},
                ],
                [{"name": "probe", "args": {"subject": "d"}}],
            ]
        )

        results = ToolCallRunner([Probe()], max_calls=1, model=model).run_turn([])

        assert len(results) == 1
        assert model.rounds == 1

    def test_the_loop_does_not_continue_forever(self):
        model = FakeModel(
            [[{"name": "probe", "args": {"subject": f"{index}"}}] for index in range(20)]
        )

        ToolCallRunner([Probe()], max_calls=20, model=model).run_turn([])

        assert model.rounds <= MAX_ROUNDS + 1

    def test_a_dropped_call_is_still_answered(self):
        """A call without an answer breaks the conversation on strict providers."""
        Probe.calls = []
        model = FakeModel(
            [
                [
                    {"name": "probe", "id": "c1", "args": {"subject": "раз"}},
                    {"name": "probe", "id": "c2", "args": {"subject": "два"}},
                ]
            ]
        )

        seen: list = []
        runner = ToolCallRunner([Probe()], max_calls=5, model=model)
        runner.model = model

        original = model.invoke

        def watch(messages):
            answer = original(messages)

            seen.extend(messages)
            return answer

        model.invoke = watch
        runner.run_turn([])

        answered = [
            message
            for message in seen
            if isinstance(message, ToolMessage)
        ]

        assert [message.tool_call_id for message in answered] == ["c1", "c2"]


class TestArguments:
    def test_the_model_chooses_the_arguments(self):
        """The old planner knew only `place`, so a brand could never be asked for."""
        Probe.calls = []
        model = FakeModel([[{"name": "probe", "args": {"subject": "бренд COS"}}]])

        ToolCallRunner([Probe()], max_calls=3, model=model).run_turn([])

        assert Probe.calls == ["бренд COS"]

    def test_arguments_the_tool_did_not_ask_for_are_ignored(self):
        Probe.calls = []
        model = FakeModel(
            [[{"name": "probe", "args": {"subject": "погода", "brand": "COS"}}]]
        )

        results = ToolCallRunner([Probe()], max_calls=3, model=model).run_turn([])

        assert results[0].ok is True
        assert Probe.calls == ["погода"]

    def test_a_tool_nobody_offered_is_refused(self):
        model = FakeModel([[{"name": "fsmtp", "args": {}}]])

        [result] = ToolCallRunner([Probe()], max_calls=3, model=model).run_turn([])

        assert result.ok is False
        assert result.error == "no such tool"


class TestReadingCalls:
    def test_calls_are_read_off_a_response(self):
        response = AIMessage(
            content="",
            tool_calls=[call("probe", "c1", subject="дождь")],
        )

        assert tools_from_response(response) == [
            {"tool": "probe", "subject": "дождь"}
        ]

    def test_a_response_with_no_calls_reads_as_nothing(self):
        assert tools_from_response(AIMessage(content="привет")) == []


class TestCheckContext:
    def test_the_node_uses_the_tools_natively(self, monkeypatch):
        from fashion_agent import tool_node

        monkeypatch.setattr(tool_node, "default_tools", lambda: [Probe()])
        Probe.calls = []

        class Model(FakeModel):
            def __init__(self):
                super().__init__([[{"name": "probe", "args": {"subject": "ветер"}}]])

        node = tool_node.check_context(
            {"messages": [], "request": {"location": "Москва"}},
            None,
            model=Model(),
        )

        assert Probe.calls == ["ветер"]
        assert node["tool_results"][0]["ok"] is True

    def test_the_node_leaves_nothing_when_nothing_was_asked(self, monkeypatch):
        from fashion_agent import tool_node

        monkeypatch.setattr(tool_node, "default_tools", lambda: [Probe()])
        node = tool_node.check_context(
            {"messages": [], "request": {}},
            None,
            model=FakeModel([[]]),
        )

        assert node == {"tool_results": []}

    def test_a_model_that_cannot_call_tools_falls_back(self, monkeypatch):
        """A turn must survive a provider that refuses tool-calling."""
        from fashion_agent import tool_node

        monkeypatch.setattr(tool_node, "default_tools", lambda: [Probe()])
        monkeypatch.setattr(
            tool_node,
            "plan_tools",
            lambda request, tools: [{"tool": "probe", "subject": "запасной"}],
        )

        class Broken(FakeModel):
            def __init__(self):
                super().__init__([[]])

            def bind_tools(self, tools):
                raise RuntimeError("tool calling not supported")

        Probe.calls = []
        node = tool_node.check_context(
            {"messages": [], "request": {}},
            None,
            model=Broken(),
        )

        assert Probe.calls == ["запасной"]
        assert node["tool_results"][0]["ok"] is True

    def test_a_turn_survives_a_broken_model_without_a_fallback(self, monkeypatch):
        from fashion_agent import tool_node

        monkeypatch.setattr(tool_node, "default_tools", lambda: [Probe()])
        monkeypatch.setattr(tool_node, "plan_tools", lambda request, tools: [])

        class Broken(FakeModel):
            def __init__(self):
                super().__init__([[]])

            def bind_tools(self, tools):
                raise RuntimeError("tool calling not supported")

        assert tool_node.check_context(
            {"messages": [], "request": {}},
            None,
            model=Broken(),
        ) == {"tool_results": []}

    def test_the_client_is_told_what_was_checked(self, monkeypatch):
        from fashion_agent import tool_node

        monkeypatch.setattr(tool_node, "default_tools", lambda: [Probe()])

        node = tool_node.check_context(
            {"messages": [], "request": {}},
            None,
            model=FakeModel([[{"name": "probe", "args": {"subject": "свет"}}]]),
        )

        assert "probe: по теме «свет» есть ответ" in node["context_lines"]


class TestRealTools:
    def test_the_weather_tool_describes_its_place(self):
        from fashion_agent.tools_weather import WeatherTool

        assert "place" in WeatherTool().args_model().model_fields

    def test_the_brand_tool_describes_its_brand(self):
        from fashion_agent.product_search.brand_reviews import BrandReviewsTool

        assert "brand" in BrandReviewsTool().args_model().model_fields

    @pytest.mark.parametrize(
        "tool_class,expected",
        [
            ("fashion_agent.tools_weather:WeatherTool", "place"),
            (
                "fashion_agent.product_search.brand_reviews:BrandReviewsTool",
                "brand",
            ),
        ],
    )
    def test_every_real_tool_can_be_offered(self, tool_class, expected):
        module, name = tool_class.split(":")
        import importlib

        tool = getattr(importlib.import_module(module), name)()
        [structured] = structured_tools([tool])

        assert expected in structured.args_schema.model_fields
