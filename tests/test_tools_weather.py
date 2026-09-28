"""The weather tool: reading, advice, and saying so when it cannot read."""

from typing import ClassVar
from urllib.parse import urlsplit

import httpx
import pytest
from fixtures import (
    GEOCODE_MOSCOW,
    GEOCODE_UNKNOWN,
    MET_NO_CLEAR,
    MET_NO_EMPTY,
    MET_NO_RAIN,
    OPEN_METEO_RAIN,
    OPEN_METEO_WARM,
    WTTR_COLD_RAIN,
)

from fashion_agent.tools import (
    ToolResult,
    ToolUnavailable,
    context_lines,
    run_tools,
    sources_ru,
)
from fashion_agent.tools_weather import (
    MetNoProvider,
    OpenMeteoProvider,
    WeatherTool,
    WttrProvider,
    describe_code,
    describe_symbol,
    geocode,
)

GEOCODE_HOST = "geocoding-api.open-meteo.com"
FORECAST_HOST = "api.open-meteo.com"
MET_HOST = "api.met.no"
WTTR_HOST = "wttr.in"

DEFAULT_RESPONSES = {
    GEOCODE_HOST: GEOCODE_MOSCOW,
    FORECAST_HOST: OPEN_METEO_RAIN,
    MET_HOST: MET_NO_CLEAR,
    WTTR_HOST: WTTR_COLD_RAIN,
}


class FakeClient:
    """Answers per host from a fixture map, and records what was asked.

    Keyed by host rather than by a substring: geocoding-api.open-meteo.com
    also contains "open-meteo", and matching loosely made one test pass for the
    wrong reason.
    """

    def __init__(self, responses: dict[str, object]):
        self.responses = responses
        self.requests: list[dict] = []
        self.closed = False

    def get(self, url, params=None, **rest):
        self.requests.append({"url": url, "params": params or {}})
        host = urlsplit(url).netloc
        response = self.responses.get(host, {})

        if isinstance(response, Exception):
            raise response

        if isinstance(response, tuple):
            status, payload = response

            return httpx.Response(status, json=payload)

        return httpx.Response(200, json=response)

    def close(self):
        self.closed = True


def weather_client(**overrides) -> FakeClient:
    return FakeClient({**DEFAULT_RESPONSES, **overrides})


def test_the_tool_says_what_it_is_for():
    tool = WeatherTool()

    assert tool.name == "weather"
    assert "погода" in tool.description.lower()
    assert tool.parameters == ["place"]


def test_a_place_is_required():
    with pytest.raises(ToolUnavailable):
        WeatherTool().run("")

    with pytest.raises(ToolUnavailable):
        WeatherTool().run(None)


def test_reads_a_forecast_from_the_first_provider():
    result = WeatherTool().run("Москва", client=weather_client())

    assert result.ok
    assert result.value["temperature_c"] == 6.1
    assert result.value["place"] == "Москва"
    assert result.source_name == "Open-Meteo"
    assert result.fetched_at


def test_the_reading_is_in_the_clients_language():
    result = WeatherTool().run("Москва", client=weather_client())
    line = result.lines[0]

    assert line.startswith("Москва:")
    assert "дождь" in line
    assert "6 °C" in line
    assert "ощущается как 1 °C" in line
    assert "осадки 80%" in line
    assert "ветер 28 м/с" in line


def test_wet_and_windy_weather_produces_wearable_advice():
    result = WeatherTool().run("Москва", client=weather_client())
    advice = " ".join(result.hints)

    assert "не боится воды" in advice
    assert "зонт" in advice
    assert "ветрено" in advice
    assert "ветрозащитный" in advice


def test_freezing_weather_asks_for_a_warm_layer():
    cold = dict(
        OPEN_METEO_RAIN,
        current={**OPEN_METEO_RAIN["current"], "temperature_2m": -4, "apparent_temperature": -9},
    )

    result = WeatherTool().run("Москва", client=weather_client(**{FORECAST_HOST: cold}))
    advice = " ".join(result.hints)

    assert "тёплый верх" in advice
    assert "верхняя одежда" in advice


def test_a_numeric_string_is_readable_as_a_temperature():
    result = WttrProvider().forecast("Москва", client=weather_client())

    assert "-2 °C" in result.lines[0]
    assert "ощущается как -6 °C" in result.lines[0]


def test_warm_weather_says_to_dress_lightly():
    client = weather_client(**{FORECAST_HOST: OPEN_METEO_WARM})
    result = WeatherTool().run("Москва", client=client)

    assert "24 °C" in result.lines[0]
    assert any("лёгкие слои" in hint for hint in result.hints)
    assert not any("зонт" in hint for hint in result.hints)


def test_a_moderate_clear_day_needs_no_advice():
    client = weather_client(**{FORECAST_HOST: OPEN_METEO_WARM, MET_HOST: MET_NO_CLEAR})
    result = WeatherTool().run("Москва", client=client)

    # 9 degrees and no rain is not worth a lecture.
    assert result.hints == [] or all("тёплый" not in hint for hint in result.hints)


def test_an_unknown_place_is_refused_rather_than_guessed():
    client = FakeClient({**DEFAULT_RESPONSES, GEOCODE_HOST: GEOCODE_UNKNOWN})

    with pytest.raises(ToolUnavailable) as error:
        WeatherTool().run("Нигде", client=client)

    assert "Нигде" in str(error.value)


def test_falls_through_to_the_next_provider():
    client = weather_client(
        **{FORECAST_HOST: httpx.ConnectTimeout("blocked")}
    )

    result = WeatherTool().run("Москва", client=client)

    assert result.source_name == "MET Norway"
    assert result.value["temperature_c"] == 9.4


def test_the_last_provider_is_tried_before_giving_up():
    client = weather_client(
        **{
            FORECAST_HOST: httpx.ConnectTimeout("blocked"),
            MET_HOST: (403, {}),
        }
    )

    result = WeatherTool().run("Москва", client=client)

    assert result.source_name == "wttr.in"
    assert result.value["temperature_c"] == -2


def test_no_provider_answering_is_a_stated_absence():
    client = weather_client(
        **{
            FORECAST_HOST: httpx.ConnectTimeout("blocked"),
            MET_HOST: httpx.ConnectTimeout("blocked"),
            WTTR_HOST: httpx.ReadTimeout("blocked"),
        }
    )

    with pytest.raises(ToolUnavailable) as error:
        WeatherTool().run("Москва", client=client)

    assert "no weather provider answered" in str(error.value)


def test_a_failing_provider_is_not_retried():
    client = weather_client(
        **{
            FORECAST_HOST: httpx.ConnectTimeout("blocked"),
            MET_HOST: (403, {}),
        }
    )
    tool = WeatherTool()

    first = tool.run("Москва", client=client)
    after_first = len(client.requests)
    second = tool.run("Москва", client=client)

    # MET Norway answers 403 here, so wttr.in is the one that does the work.
    assert first.source_name == "wttr.in"
    assert first.value["temperature_c"] == -2
    assert second.value["temperature_c"] == -2
    # The reading is cached, so a repeat costs no request at all.
    assert len(client.requests) == after_first


def test_a_repeated_question_is_answered_from_cache():
    client = weather_client()
    tool = WeatherTool()

    tool.run("Москва", client=client)
    tool.run("  москва  ", client=client)

    geocodes = [request for request in client.requests if request["url"] and GEOCODE_HOST in request["url"]]

    assert len(geocodes) == 1


def test_a_cache_expires():
    client = weather_client()
    tool = WeatherTool()

    tool.run("Москва", client=client)
    tool._cache[("weather", "москва")] = (0.0, tool._cache[("weather", "москва")][1])
    tool.run("Москва", client=client)

    geocodes = [request for request in client.requests if request["url"] and GEOCODE_HOST in request["url"]]

    assert len(geocodes) == 2


def test_a_met_no_symbol_becomes_russian():
    result = MetNoProvider().forecast(55.75, 37.62, client=FakeClient({**DEFAULT_RESPONSES, MET_HOST: MET_NO_RAIN}))

    assert "дождь" in result.lines[0]
    assert any("не боится воды" in hint for hint in result.hints)
    assert "80%" in result.lines[0]


def test_met_no_without_a_reading_is_unavailable():
    with pytest.raises(ToolUnavailable):
        MetNoProvider().forecast(
            55.75,
            37.62,
            client=FakeClient({**DEFAULT_RESPONSES, MET_HOST: MET_NO_EMPTY}),
        )


def test_wttr_uses_the_translated_description():
    result = WttrProvider().forecast("Москва", client=FakeClient({**DEFAULT_RESPONSES, WTTR_HOST: WTTR_COLD_RAIN}))

    assert "Небольшой дождь" in result.lines[0]
    assert any("тёплый верх" in hint for hint in result.hints)


def test_open_meteo_without_a_reading_is_unavailable():
    with pytest.raises(ToolUnavailable):
        OpenMeteoProvider().forecast(
            55.75,
            37.62,
            client=FakeClient({**DEFAULT_RESPONSES, FORECAST_HOST: {"current": {}}}),
        )


def test_geocode_returns_coordinates():
    place = geocode("Москва", client=weather_client())

    assert place["name"] == "Москва"
    assert place["latitude"] == pytest.approx(55.75204)
    assert place["source_url"]


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("clearsky_day", ("ясно", None)),
        ("partlycloudy_night", ("переменная облачность", None)),
        ("clearsky_polartwilight", ("ясно", None)),
        ("heavyrain", ("сильный дождь", "rain")),
        ("rainshowers_day", ("ливень", "rain")),
        ("lightsnow", ("небольшой снег", "snow")),
        ("thunder", ("гроза", "storm")),
        ("", ("погода неизвестна", None)),
        ("something_else", ("погода неизвестна", None)),
    ],
)
def test_describe_symbol(symbol, expected):
    assert describe_symbol(symbol) == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (0, ("ясно", None)),
        (61, ("небольшой дождь", "rain")),
        (71, ("небольшой снег", "snow")),
        (95, ("гроза", "storm")),
        (45, ("туман", "visibility")),
        (None, ("погода неизвестна", None)),
    ],
)
def test_describe_code(code, expected):
    assert describe_code(code) == expected


def test_the_planner_sees_the_catalogue():
    from fashion_agent.tools import catalogue

    listing = catalogue([WeatherTool()])

    assert "weather(place)" in listing
    assert "Погода" in listing


def test_the_same_tool_is_not_run_twice_in_a_turn():
    # The tool builds its own client here, so this only checks the dedupe.
    results = run_tools(
        [WeatherTool()],
        [
            {"tool": "weather", "place": "Москва"},
            {"tool": "weather", "place": "Казань"},
        ],
    )

    assert len(results) == 1
    assert results[0].ok


def test_an_unknown_tool_is_reported():
    results = run_tools([WeatherTool()], [{"tool": "stocks", "place": "Москва"}])

    assert results[0].ok is False
    assert results[0].error == "no such tool"


def test_too_many_tool_calls_are_cut():
    class Counting:
        description = "counts"
        parameters: ClassVar[list[str]] = []

        def available(self):
            return True

        def run(self, **params):
            return ToolResult(tool=self.name, lines=["ок"])

    tools = [type("Counting", (Counting,), {"name": f"tool_{index}"})() for index in range(6)]

    results = run_tools(tools, [{"tool": tool.name} for tool in tools])

    assert len(results) == 3


def test_a_raising_tool_does_not_kill_the_turn():
    class Boom:
        name = "boom"
        description = "raises"
        parameters: ClassVar[list[str]] = []

        def available(self):
            return True

        def run(self, **params):
            raise RuntimeError("kaboom")

    results = run_tools([Boom()], [{"tool": "boom"}])

    assert results[0].ok is False
    assert "RuntimeError" in results[0].error


def test_an_unavailable_tool_is_reported_not_skipped():
    class Off:
        name = "off"
        description = "not configured"
        parameters: ClassVar[list[str]] = []

        def available(self):
            return False

        def run(self, **params):
            raise AssertionError("must not run")

    results = run_tools([Off()], [{"tool": "off"}])

    assert results[0].ok is False
    assert "not configured" in results[0].error


def test_a_fact_carries_its_source():
    client = weather_client()
    result = WeatherTool().run("Москва", client=client)

    assert result.source_line.startswith("Open-Meteo — ")


def test_the_client_is_told_about_sources_and_gaps():
    results = [
        ToolResult(
            tool="weather",
            ok=True,
            lines=["Москва: дождь, 6 °C"],
            source_name="Open-Meteo",
            source_url="https://open-meteo.com/",
        ),
        ToolResult(tool="trends", ok=False, error="not configured"),
    ]

    assert "Open-Meteo" in sources_ru(results)[0]
    assert "проверить не удалось" in sources_ru(results)[1]


def test_context_lines_feed_the_prompts():
    results = [
        ToolResult(
            tool="weather",
            ok=True,
            lines=["Москва: дождь, 6 °C"],
            hints=["обувь, которая не боится воды"],
        )
    ]

    joined = " ".join(context_lines(results))

    assert "дождь, 6 °C" in joined
    assert "не боится воды" in joined


def test_a_failed_tool_adds_nothing_to_the_prompts():
    results = [ToolResult(tool="weather", ok=False, error="down")]

    assert context_lines(results) == []


def test_a_client_the_caller_owns_is_left_open():
    client = weather_client()
    tool = WeatherTool()
    tool.run("Москва", client=client)

    # A client passed in belongs to the caller, so it must be left open.
    assert client.closed is False
