"""Tools the agent may reach for, and the record of what it got.

A tool declares a description and the parameters it needs, and the agent chooses
which to run. Every result carries where it came from and when, so a reply can
tell the client what was just checked and what is the agent's own knowledge, and
so an unavailable tool leaves a stated absence rather than a confident guess.

Tool runs are bounded per turn. A request for an outfit is not a licence to
crawl, and a plan that keeps asking for more is a bug, not persistence.
"""

from datetime import UTC, datetime
from typing import Any, Protocol

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# A turn should not turn into a crawl.
MAX_TOOLS_PER_TURN = 3
DEFAULT_TIMEOUT = 12.0

# A cached answer may be reused for this long, which is enough to survive a
# refresh storm and short enough that the forecast is still a forecast.
CACHE_TTL_SECONDS = 900


class ToolUnavailable(Exception):
    """The tool is not configured, or could not be reached."""


class ToolResult(BaseModel):
    tool: str
    ok: bool = True
    value: dict[str, Any] = Field(default_factory=dict)
    lines: list[str] = Field(
        default_factory=list,
        description="What to tell the client, in their language",
    )
    hints: list[str] = Field(
        default_factory=list,
        description="Consequences for the outfit, as plain sentences",
    )
    source_url: str | None = None
    source_name: str | None = None
    fetched_at: str = ""
    error: str | None = None

    @property
    def source_line(self) -> str:
        if self.source_name and self.source_url:
            return f"{self.source_name} — {self.source_url}"

        return self.source_name or self.source_url or "источник неизвестен"


def now() -> str:
    return datetime.now(UTC).isoformat()


class Tool(Protocol):
    name: str
    description: str
    parameters: list[str]

    def args_model(self) -> type[BaseModel]:
        """The parameters, typed.

        The agent is handed this rather than a hand-written list, so a tool with
        a new argument needs no change anywhere else: adding a field here is the
        only edit a new function requires.
        """
        ...

    def available(self) -> bool: ...

    def run(self, **params: Any) -> ToolResult: ...


class NoArguments(BaseModel):
    """A tool that takes nothing."""


def as_structured_tool(tool: Tool):
    """A tool the model can call natively, schema and all."""
    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_model(),
        func=lambda **params: tool.run(**params).model_dump(mode="json"),
    )


def structured_tools(tools: list[Tool]) -> list:
    """The tools the model may call, skipping the ones that cannot answer.

    Offering a tool that is known to be unconfigured wastes a turn on a refusal
    the model could have predicted.
    """
    return [as_structured_tool(tool) for tool in tools if tool.available()]


def catalogue(tools: list[Tool]) -> str:
    """The tool list, as the planner sees it."""
    lines = []

    for tool in tools:
        parameters = ", ".join(tool.parameters) if tool.parameters else "ничего"

        lines.append(
            f"- {tool.name}({parameters}): {tool.description} "
            f"[{'доступен' if tool.available() else 'сейчас недоступен'}]"
        )

    return "\n".join(lines) or "- инструментов нет"


def run_tools(
    tools: list[Tool],
    calls: list[dict],
) -> list[ToolResult]:
    """Run the requested tools, once each, in the order the agent asked."""
    results: list[ToolResult] = []
    seen: set[str] = set()
    available = {tool.name: tool for tool in tools}

    for call in calls[:MAX_TOOLS_PER_TURN]:
        name = str(call.get("tool") or "").strip()

        if not name or name in seen:
            continue

        seen.add(name)
        tool = available.get(name)

        if tool is None:
            results.append(
                ToolResult(
                    tool=name,
                    ok=False,
                    error="no such tool",
                    fetched_at=now(),
                )
            )
            continue

        if not tool.available():
            results.append(
                ToolResult(
                    tool=name,
                    ok=False,
                    error="tool is not configured or is not answering",
                    fetched_at=now(),
                )
            )
            continue

        params = {
            key: value
            for key, value in call.items()
            if key != "tool" and value not in (None, "")
        }

        try:
            result = tool.run(**params)
        except ToolUnavailable as error:
            results.append(
                ToolResult(
                    tool=name,
                    ok=False,
                    error=str(error),
                    fetched_at=now(),
                )
            )
            continue
        except Exception as error:  # noqa: BLE001 - a tool must not kill a turn
            results.append(
                ToolResult(
                    tool=name,
                    ok=False,
                    error=f"{type(error).__name__}: {error}",
                    fetched_at=now(),
                )
            )
            continue

        if not result.fetched_at:
            result.fetched_at = now()

        results.append(result)

    return results


def context_lines(results: list[ToolResult]) -> list[str]:
    """Lines for a prompt: what was checked, and what to do about it."""
    lines: list[str] = []

    for result in results:
        if not result.ok:
            continue

        lines.append(f"{result.tool}: " + "; ".join(result.lines))

        if result.hints:
            lines.extend(f"{result.tool}: " + hint for hint in result.hints)

    return lines


def sources_ru(results: list[ToolResult]) -> list[str]:
    """What to show the client, including the tools that failed."""
    lines: list[str] = []

    for result in results:
        if result.ok:
            lines.append(f"{result.tool}: {result.source_line}")
        else:
            lines.append(
                f"{result.tool}: проверить не удалось"
                + (f" ({result.error})" if result.error else "")
            )

    return lines


def hints(results: list[ToolResult]) -> list[str]:
    collected: list[str] = []

    for result in results:
        for hint in result.hints:
            if hint not in collected:
                collected.append(hint)

    return collected
