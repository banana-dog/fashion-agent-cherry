"""The agent calling tools itself, rather than filling in a form about them.

The earlier arrangement asked the model to describe its tool calls in a
structured answer. That is not the same as tool-calling: the model had to know
the shape of every argument in advance, and the shape was written out by hand in
one place. The moment a tool took a new argument, that list was wrong, and
nothing said so.

Here the tools are handed to the model as tools. Their arguments come from the
tools themselves, so a new function needs no change anywhere else, and the model
cannot name an argument that does not exist.

The loop is bounded three ways on purpose. A turn is a request for an outfit, not
a licence to browse: at most a few calls in total, at most a couple of rounds,
and a repeated call is dropped rather than repeated. A tool that fails or refuses
returns that as its answer, so the loop still ends.
"""

from langchain_core.messages import (
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool

from fashion_agent.llm import llm
from fashion_agent.tools import (
    ToolResult,
    run_tools,
    structured_tools,
)

# One extra call ends the loop: the model is asked what else it needs, and an
# answer with no tool calls means it is done.
MAX_ROUNDS = 2

# What the model is told before it is offered anything.
CALLER_PROMPT = """
You can call outside tools to check facts before you answer a fashion client.

Rules:
- Call a tool only when the answer would change what you say. A forecast matters
  for what to wear outside; it does not matter for a question about colour.
- Never state a fact you did not get from a tool. If a tool fails, say that you
  could not check it.
- Prefer not calling anything. Deciding you do not need a tool is a good answer.
- Use the city the client gave. Never invent one.
- Give a brand exactly as the client named it.
- At most one call per tool, and never the same call twice.
"""


class ToolCallRunner:
    """Runs the tools the model asks for, within a budget."""

    def __init__(
        self,
        tools: list,
        *,
        max_calls: int,
        model=None,
    ):
        self.tools = tools
        self.max_calls = max_calls
        self.model = model if model is not None else llm
        self.spent = 0
        self.seen: set[tuple] = set()

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.max_calls

    def _spend(self, count: int) -> None:
        self.spent += count

    def run_turn(
        self,
        messages: list,
        *,
        rounds: int = MAX_ROUNDS,
    ) -> list[ToolResult]:
        """Let the model ask for what it needs, and collect the answers."""
        available = structured_tools(self.tools)

        if not available:
            return []

        bound = self.model.bind_tools(available)
        history = [SystemMessage(content=CALLER_PROMPT), *messages]
        results: list[ToolResult] = []

        for _round in range(max(1, rounds)):
            response = bound.invoke(history)

            emitted = [
                call
                for call in (getattr(response, "tool_calls", None) or [])
                if isinstance(call, dict)
            ]
            wanted = self._wanted(emitted)

            if not wanted:
                break

            answers = self._answer_each(emitted, wanted)

            spent = sum(1 for _c, result, _s in answers if result is not None)
            self._spend(spent)
            results.extend(
                result for _call, result, _skipped in answers if result is not None
            )

            history = [
                *history,
                response,
                *[
                    ToolMessage(
                        content=_answer(result, skipped),
                        tool_call_id=str(call.get("id") or ""),
                    )
                    for call, result, skipped in answers
                ],
            ]

            if self.exhausted:
                break

        return results

    def _answer_each(
        self,
        emitted: list[dict],
        wanted: list[tuple[int, dict]],
    ) -> list[tuple[dict, ToolResult | None, bool]]:
        """Pair every call the model made with an answer to it.

        The assistant message still contains all of them, so every one gets a
        reply. A provider sent a tool call without its answer rejects the whole
        conversation, and a turn that died here would cost the client their outfit
        over a de-duplication detail.
        """
        run_results = run_tools(self.tools, [call for _position, call in wanted])
        results_by_position = {
            position: result
            for (position, _call), result in zip(wanted, run_results, strict=False)
        }
        answers: list[tuple[dict, ToolResult | None, bool]] = []

        for position, call in enumerate(emitted):
            answers.append(
                (call, results_by_position.get(position), position not in results_by_position)
            )

        return answers

    def _wanted(self, emitted: list[dict]) -> list[tuple[int, dict]]:
        """Which calls to run, as (position in the reply, call) pairs.

        Positions rather than ids, because a provider is free to reuse an id and
        pairing by it would then hand the same answer to two different calls.
        """
        wanted: list[tuple[int, dict]] = []
        names: set[str] = set()
        room = max(0, self.max_calls - self.spent)

        for position, call in enumerate(emitted):
            if len(wanted) >= room:
                break

            name = str(call.get("name") or "")
            arguments = call.get("args") or {}

            if not isinstance(arguments, dict):
                arguments = {}

            key = (
                name,
                tuple(sorted((str(k), str(v)) for k, v in arguments.items())),
            )

            # A repeated call, or a second call to a tool already used this turn,
            # is answered but not run again.
            if key in self.seen or name in names:
                continue

            self.seen.add(key)
            names.add(name)
            wanted.append((position, {"tool": name, **arguments}))

        return wanted

    @property
    def used(self) -> int:
        return self.spent


def _answer(
    result: ToolResult | None,
    skipped: bool = False,
) -> str:
    """What the model is told after a tool ran.

    A failure is reported as a failure. Reporting it as an empty answer would let
    the model fill the gap from its own memory, which is exactly the guess the
    tool was called to prevent.
    """
    if result is None:
        return (
            "Повторять этот вызов не нужно: ответ уже получен "
            "или лимит на этот ход исчерпан."
            if skipped
            else "Вызов не выполнен."
        )

    if not result.ok:
        return f"Инструмент недоступен: {result.error or 'причина неизвестна'}"

    if not result.lines:
        return "Инструмент отработал, но ничего не вернул."

    return "\n".join([*result.lines, f"Источник: {result.source_line}"])


def tools_from_response(
    response,
) -> list[dict]:
    """The calls a model response asked for, as plain dicts."""
    calls = []

    for call in getattr(response, "tool_calls", None) or []:
        if not isinstance(call, dict):
            continue

        arguments = call.get("args")

        calls.append(
            {
                "tool": str(call.get("name") or ""),
                **(
                    arguments
                    if isinstance(arguments, dict)
                    else {}
                ),
            }
        )

    return calls


def describe(tools: list) -> str:
    """The tools, spelled out, for prompts and for the record."""
    lines = []

    for tool in tools:
        schema = tool.args_model()
        fields = ", ".join(
            f"{name}: {field.description or 'без описания'}"
            for name, field in schema.model_fields.items()
        )
        state = "доступен" if tool.available() else "сейчас недоступен"

        lines.append(f"- {tool.name}({fields}) — {state}")

    return "\n".join(lines) or "- инструментов нет"


__all__ = [
    "MAX_ROUNDS",
    "StructuredTool",
    "ToolCallRunner",
    "describe",
    "tools_from_response",
]
