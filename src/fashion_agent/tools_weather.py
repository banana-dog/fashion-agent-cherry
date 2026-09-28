"""Weather, as a tool.

A forecast is only useful here if it changes what the agent suggests, so a
result carries both the reading in words and the consequences: a coat, boots
that keep the feet dry, an umbrella. A location the client gave is treated as
the place they are in; guessing one is not an option, so an unknown city is a
stated absence.

Providers are tried in order and the first to answer wins. On a network where
none answers, the tool says so and the agent says so, which is the point: a
fabricated forecast is worse than none.
"""

import os
import time
from typing import ClassVar

import httpx

from fashion_agent.tools import (
    CACHE_TTL_SECONDS,
    DEFAULT_TIMEOUT,
    ToolResult,
    ToolUnavailable,
    now,
)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
MET_NO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
WTTR_URL = "https://wttr.in/{place}"

COLD_LIMIT = 5
WARM_LIMIT = 18
RAIN_LIMIT = 55
WIND_LIMIT = 25.0

# WMO weather codes, grouped by what a person has to do about them.
WEATHER_CODES = {
    0: ("ясно", None),
    1: ("почти ясно", None),
    2: ("переменная облачность", None),
    3: ("пасмурно", "light"),
    45: ("туман", "visibility"),
    48: ("изморозь", "visibility"),
    51: ("морось", "rain"),
    53: ("морось", "rain"),
    55: ("сильная морось", "rain"),
    56: ("ледяная морось", "rain"),
    57: ("ледяная морось", "rain"),
    61: ("небольшой дождь", "rain"),
    63: ("дождь", "rain"),
    65: ("сильный дождь", "rain"),
    66: ("ледяной дождь", "rain"),
    67: ("ледяной дождь", "rain"),
    71: ("небольшой снег", "snow"),
    73: ("снег", "snow"),
    75: ("сильный снег", "snow"),
    77: ("снежная крупа", "snow"),
    80: ("ливень", "rain"),
    81: ("ливень", "rain"),
    82: ("сильный ливень", "rain"),
    85: ("снежный заряд", "snow"),
    86: ("сильный снежный заряд", "snow"),
    95: ("гроза", "storm"),
    96: ("гроза с градом", "storm"),
    99: ("гроза с градом", "storm"),
}


# MET Norway reports a symbol code rather than a WMO number, with a
# day/night suffix that says nothing about the weather.
SYMBOL_CODES = {
    "clearsky": ("ясно", None),
    "fair": ("малооблачно", None),
    "partlycloudy": ("переменная облачность", None),
    "cloudy": ("пасмурно", "light"),
    "fog": ("туман", "visibility"),
    "lightrain": ("небольшой дождь", "rain"),
    "rain": ("дождь", "rain"),
    "heavyrain": ("сильный дождь", "rain"),
    "lightrainshowers": ("небольшой ливень", "rain"),
    "rainshowers": ("ливень", "rain"),
    "heavyrainshowers": ("сильный ливень", "rain"),
    "lightsleet": ("мокрый снег", "rain"),
    "sleet": ("снег с дождём", "rain"),
    "heavysleet": ("сильный снег с дождём", "rain"),
    "lightsnow": ("небольшой снег", "snow"),
    "snow": ("снег", "snow"),
    "heavysnow": ("сильный снег", "snow"),
    "lightsnowshowers": ("снежный дождь", "snow"),
    "snowshowers": ("снегопад", "snow"),
    "heavysnowshowers": ("сильный снегопад", "snow"),
    "thunder": ("гроза", "storm"),
}


def describe_code(code: int | None) -> tuple[str, str | None]:
    # A missing code must not read as a clear sky: that is the most misleading
    # thing to say about weather nobody reported.
    if code is None or code == "":
        return ("погода неизвестна", None)

    try:
        return WEATHER_CODES.get(int(code), ("погода неизвестна", None))
    except (TypeError, ValueError):
        return ("погода неизвестна", None)


def describe_symbol(symbol: str | None) -> tuple[str, str | None]:
    if not symbol:
        return ("погода неизвестна", None)

    name = symbol.strip().lower()

    for suffix in ("_polartwilight", "_day", "_night", "_twilight"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    return SYMBOL_CODES.get(name, ("погода неизвестна", None))


def geocode(place: str, *, client: httpx.Client | None = None) -> dict:
    """Find a place by name, so a forecast needs no coordinates from the user."""
    text = place.strip()

    if not text:
        raise ToolUnavailable("no place to look up")

    own = client is None
    session = client or httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True)

    try:
        response = session.get(
            GEOCODE_URL,
            params={
                "name": text,
                "count": 1,
                "language": "ru",
                "format": "json",
            },
        )
    except httpx.HTTPError as error:
        raise ToolUnavailable(f"geocoding failed: {type(error).__name__}") from error
    finally:
        if own:
            session.close()

    results = response.json().get("results") or []

    if not results:
        raise ToolUnavailable(f"could not find a place called {text!r}")

    first = results[0]

    return {
        "name": first.get("name") or text,
        "region": first.get("admin1") or "",
        "country": first.get("country") or "",
        "latitude": first["latitude"],
        "longitude": first["longitude"],
        "source_url": "https://open-meteo.com/en/docs/geocoding-api",
    }


def _hint_lines(
    temperature: float | None,
    feels_like: float | None,
    code_kind: str | None,
    precipitation_chance: int | None,
    wind: float | None,
) -> list[str]:
    hints: list[str] = []

    if temperature is not None:
        if temperature <= COLD_LIMIT:
            hints.append(
                f"На улице {temperature:.0f} °C: нужен тёплый верх "
                "и, скорее всего, верхняя одежда."
            )
        elif temperature >= WARM_LIMIT:
            hints.append(
                f"На улице {temperature:.0f} °C: лёгкие слои, "
                "без тяжёлой верхней одежды."
            )

    if feels_like is not None and temperature is not None and feels_like <= temperature - 4:
        hints.append(
            f"Ощущается как {feels_like:.0f} °C: ветрено, "
            "ветрозащитный слой важнее утепления."
        )

    if code_kind in {"rain", "storm"}:
        hints.append("Осадки: обувь, которая не боится воды, и зонт.")

    if code_kind == "snow":
        hints.append("Снег: обувь с протектором, лучше не скользить.")

    if code_kind == "visibility":
        hints.append("Плохая видимость: верхний контрастный элемент, чтобы было видно.")

    if wind is not None and wind >= WIND_LIMIT:
        hints.append(f"Ветер {wind:.0f} м/с: прямой силуэт лучше, чем развевающийся.")

    if precipitation_chance is not None and precipitation_chance >= RAIN_LIMIT:
        hints.append(f"Вероятность осадков до {precipitation_chance}%.")

    return hints


def _as_number(value) -> float | None:
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finish(
    *,
    temperature: float | None,
    feels_like: float | None,
    code_kind: str | None,
    description: str,
    precipitation_chance: int | None,
    wind: float | None,
    source_name: str,
    source_url: str,
) -> ToolResult:
    # Providers disagree on types: MET Norway sends numbers, wttr.in sends
    # strings, and a format spec on a string is a crash.
    temperature = _as_number(temperature)
    feels_like = _as_number(feels_like)
    wind = _as_number(wind)

    if precipitation_chance is not None:
        try:
            precipitation_chance = int(precipitation_chance)
        except (TypeError, ValueError):
            precipitation_chance = None

    line = description

    if temperature is not None:
        line += f", {temperature:.0f} °C"

        if feels_like is not None:
            line += f" (ощущается как {feels_like:.0f} °C)"

    if precipitation_chance is not None:
        line += f", осадки {precipitation_chance}%"

    if wind is not None:
        line += f", ветер {wind:.0f} м/с"

    return ToolResult(
        tool="weather",
        lines=[line],
        hints=_hint_lines(
            temperature,
            feels_like,
            code_kind,
            precipitation_chance,
            wind,
        ),
        value={
            "temperature_c": temperature,
            "apparent_c": feels_like,
            "description": description,
            "precipitation_chance": precipitation_chance,
            "wind_speed": wind,
        },
        source_name=source_name,
        source_url=source_url,
    )


class OpenMeteoProvider:
    name = "open-meteo"

    def available(self) -> bool:
        return True

    def forecast(
        self,
        latitude: float,
        longitude: float,
        *,
        client: httpx.Client | None = None,
    ) -> ToolResult:
        own = client is None
        session = client or httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True)

        try:
            response = session.get(
                OPEN_METEO_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": (
                        "temperature_2m,apparent_temperature,"
                        "precipitation,wind_speed_10m,weather_code"
                    ),
                    "hourly": "precipitation_probability",
                    "forecast_days": 1,
                    "timezone": "auto",
                },
            )
        except httpx.HTTPError as error:
            raise ToolUnavailable(
                f"{self.name} did not answer: {type(error).__name__}"
            ) from error
        finally:
            if own:
                session.close()

        data = response.json()
        current = data.get("current") or {}

        if "temperature_2m" not in current:
            raise ToolUnavailable(f"{self.name} returned no reading")

        hourly = data.get("hourly") or {}
        chances = [
            value
            for value in (hourly.get("precipitation_probability") or [])
            if isinstance(value, (int, float))
        ]

        description, kind = describe_code(current.get("weather_code"))

        return _finish(
            temperature=current.get("temperature_2m"),
            feels_like=current.get("apparent_temperature"),
            code_kind=kind,
            description=description,
            precipitation_chance=max(chances) if chances else None,
            wind=current.get("wind_speed_10m"),
            source_name="Open-Meteo",
            source_url="https://open-meteo.com/",
        )


class MetNoProvider:
    name = "met.no"

    def available(self) -> bool:
        return True

    def forecast(
        self,
        latitude: float,
        longitude: float,
        *,
        client: httpx.Client | None = None,
    ) -> ToolResult:
        own = client is None
        session = client or httpx.Client(
            timeout=DEFAULT_TIMEOUT,
            follow_redirects=True,
            headers={
                "User-Agent": os.getenv("CHERRY_USER_AGENT", "CherryPick/0.1"),
            },
        )

        try:
            response = session.get(
                MET_NO_URL,
                params={"lat": latitude, "lon": longitude},
            )
        except httpx.HTTPError as error:
            raise ToolUnavailable(
                f"{self.name} did not answer: {type(error).__name__}"
            ) from error
        finally:
            if own:
                session.close()

        if response.status_code != 200:
            raise ToolUnavailable(f"{self.name} replied {response.status_code}")

        timeseries = (response.json().get("properties") or {}).get("timeseries") or []

        if not timeseries:
            raise ToolUnavailable(f"{self.name} returned no reading")

        details = ((timeseries[0].get("data") or {}).get("instant") or {}).get(
            "details"
        ) or {}
        following = (timeseries[0].get("data") or {}).get("next_1_hours") or {}

        symbol = (following.get("summary") or {}).get("symbol_code") or ""
        description, kind = describe_symbol(symbol)

        return _finish(
            temperature=details.get("air_temperature"),
            feels_like=None,
            code_kind=kind,
            description=description,
            precipitation_chance=following.get("details", {}).get(
                "probability_of_precipitation"
            ),
            wind=details.get("wind_speed"),
            source_name="MET Norway",
            source_url="https://api.met.no/doc/Weatherapi",
        )


class WttrProvider:
    name = "wttr.in"

    def available(self) -> bool:
        return True

    def forecast(
        self,
        place: str,
        *,
        client: httpx.Client | None = None,
    ) -> ToolResult:
        own = client is None
        session = client or httpx.Client(
            timeout=DEFAULT_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "curl/8"},
        )

        try:
            response = session.get(
                WTTR_URL.format(place=place),
                params={"format": "j1", "lang": "ru"},
            )
        except httpx.HTTPError as error:
            raise ToolUnavailable(
                f"{self.name} did not answer: {type(error).__name__}"
            ) from error
        finally:
            if own:
                session.close()

        conditions = response.json().get("current_condition") or []

        if not conditions:
            raise ToolUnavailable(f"{self.name} returned no reading")

        current = conditions[0]
        description = (current.get("lang_ru") or [{}])[0].get("value") or (
            current.get("weatherDesc") or [{}]
        )[0].get("value", "")

        text = description.lower()
        kind = (
            "rain"
            if any(word in text for word in ("дожд", "ливень", "морось"))
            else (
                "snow"
                if any(word in text for word in ("снег", "снегопад"))
                else ("storm" if "гроз" in text else None)
            )
        )

        return _finish(
            temperature=current.get("temp_C"),
            feels_like=current.get("FeelsLikeC"),
            code_kind=kind,
            description=description,
            precipitation_chance=None,
            wind=current.get("windspeedKmph"),
            source_name="wttr.in",
            source_url=f"https://wttr.in/{place}",
        )


PROVIDERS = (OpenMeteoProvider(), MetNoProvider(), WttrProvider())


class WeatherTool:
    """A forecast for a named place, from whichever provider answers first."""

    name = "weather"
    description = (
        "Погода в городе клиента: температура, ощущается как, осадки, ветер. "
        "Нужна, чтобы не предлагать шубу в мае и зонт там, где сухо. "
        "Бери только когда в запросе есть или город, или явная погода."
    )
    parameters: ClassVar[list[str]] = ["place"]

    def __init__(
        self,
        providers: tuple = PROVIDERS,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        sleeper=time.sleep,
    ):
        self.providers = providers
        self.timeout = timeout
        self.sleeper = sleeper
        self._cache: dict[tuple, tuple[float, ToolResult]] = {}
        self._failed: set[str] = set()

    def available(self) -> bool:
        return bool(self.providers)

    def _cached(self, key: tuple) -> ToolResult | None:
        entry = self._cache.get(key)

        if entry is None:
            return None

        saved_at, result = entry

        if time.time() - saved_at > CACHE_TTL_SECONDS:
            return None

        return result

    def _remember(self, key: tuple, result: ToolResult) -> None:
        if len(self._cache) >= 32:
            self._cache.pop(next(iter(self._cache)))

        self._cache[key] = (time.time(), result)

    def run(
        self,
        place: str | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> ToolResult:
        text = (place or "").strip()

        if not text:
            raise ToolUnavailable("no place given, so there is nothing to look up")

        key = ("weather", text.casefold())
        cached = self._cached(key)

        if cached is not None:
            return cached

        session = client or httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
        )
        own = client is None
        errors: list[str] = []

        try:
            place_info = geocode(text, client=session)

            for provider in self.providers:
                if provider.name in self._failed:
                    errors.append(f"{provider.name}: already failing")

                    continue

                try:
                    if provider.name == "wttr.in":
                        result = provider.forecast(text, client=session)
                    else:
                        result = provider.forecast(
                            place_info["latitude"],
                            place_info["longitude"],
                            client=session,
                        )
                except ToolUnavailable as error:
                    errors.append(str(error))
                    self._failed.add(provider.name)
                    continue

                place_name = place_info["name"]
                result.value["place"] = place_name
                result.lines = [f"{place_name}: {line}" for line in result.lines]
                result.fetched_at = now()
                self._remember(key, result)

                return result

            raise ToolUnavailable(
                "no weather provider answered: " + "; ".join(errors[:3])
            )
        finally:
            if own:
                session.close()

