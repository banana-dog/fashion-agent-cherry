"""What the agent did, and whether it was any good.

Two different questions get answered here, and they are easy to confuse. A trace
records what happened in one turn, so that anybody looking at a strange answer
can see the steps that produced it. Metrics count what happens across many
turns, so that a slow drift in quality is visible before a client reports it.

Both stay inside the process. That is a deliberate limit rather than an
oversight: this is one script on one machine, the records are small, and
shipping them to a service would add a dependency the project does not have. The
counters are bounded so a long-running process cannot grow without limit.

A metric nobody can act on is decoration, so each one carries the question it
answers rather than just a number. "Kept 12 of 40" says nothing; "size filters
dropped half of what came back" says everything.
"""

import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

# Long enough to see a trend, short enough to forget a bad week.
WINDOW_SECONDS = 7 * 24 * 3600

MAX_TRACES = 50

MAX_EVENTS_PER_KIND = 500


@dataclass
class Sample:
    """One counted thing, with when it happened so the window can slide."""

    at: float
    fields: dict[str, Any] = field(default_factory=dict)


class Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, list[Sample]] = defaultdict(list)
        self._traces: list[dict] = []

    def record(self, kind: str, **fields: Any) -> None:
        with self._lock:
            events = self._events[kind]
            events.append(Sample(time.time(), fields))

            if len(events) > MAX_EVENTS_PER_KIND:
                del events[: len(events) - MAX_EVENTS_PER_KIND]

    def _window(
        self,
        kind: str,
        seconds: float = WINDOW_SECONDS,
    ) -> list[Sample]:
        cutoff = time.time() - seconds

        return [sample for sample in self._events.get(kind, []) if sample.at >= cutoff]

    # search

    def search_run(
        self,
        *,
        source: str,
        ok: bool,
        raw: int,
        kept: int,
        latency_ms: int,
        relaxed: list[str] | None = None,
        filtered_out: dict[str, int] | None = None,
        category: str | None = None,
    ) -> None:
        self.record(
            "search",
            source=source,
            ok=ok,
            raw=raw,
            kept=kept,
            latency_ms=latency_ms,
            relaxed=list(relaxed or []),
            filtered_out=dict(filtered_out or {}),
            category=category,
        )

    def tool_run(
        self,
        *,
        tool: str,
        ok: bool,
        latency_ms: int = 0,
    ) -> None:
        self.record("tool", tool=tool, ok=ok, latency_ms=latency_ms)

    def knowledge_use(
        self,
        *,
        formulas: list[str],
        trends: list[str],
        used_items: int,
        bought_items: int,
    ) -> None:
        self.record(
            "knowledge",
            formulas=list(formulas or []),
            trends=list(trends or []),
            used_items=used_items,
            bought_items=bought_items,
        )

    def turn(
        self,
        *,
        threads: int,
        duration_ms: int,
        outfits: int,
        had_error: bool,
    ) -> None:
        self.record(
            "turn",
            threads=threads,
            duration_ms=duration_ms,
            outfits=outfits,
            had_error=had_error,
        )

    # traces

    def start_trace(
        self,
        *,
        user_id: str,
        thread_id: str,
        question: str,
    ) -> "Trace":
        trace = Trace(user_id=user_id, thread_id=thread_id, question=question)
        self.add_trace(trace)

        return trace

    def add_trace(self, trace: "Trace") -> None:
        with self._lock:
            self._traces.append(trace.as_dict())
            self._trim_traces()

    def update_trace(self, trace: "Trace") -> None:
        """Rewrite a trace that is still running.

        A trace is registered the moment a turn starts, so that a turn which dies
        is still visible, and updated when it finishes. Registering it twice
        would make one turn look like two, which is the sort of small wrong
        number that makes a whole page untrustworthy.
        """
        with self._lock:
            for index, stored in enumerate(self._traces):
                if stored.get("id") == trace.id:
                    self._traces[index] = trace.as_dict()
                    return

            self._traces.append(trace.as_dict())
            self._trim_traces()

    def _trim_traces(self) -> None:
        if len(self._traces) > MAX_TRACES:
            del self._traces[: len(self._traces) - MAX_TRACES]

    def traces(self, user_id: str | None = None) -> list[dict]:
        with self._lock:
            found = list(self._traces)

        if user_id is not None:
            found = [trace for trace in found if trace.get("user_id") == user_id]

        return list(reversed(found))

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._traces.clear()


class Trace:
    """One turn, step by step, for somebody asking "why did it say that?".

    Recorded but never shown to the client: it names the constraints that were
    given up and the searches that failed, which is exactly what a stylist
    should know and a person being sold a coat should not.
    """

    def __init__(self, *, user_id: str, thread_id: str, question: str):
        self.user_id = user_id
        self.thread_id = thread_id
        self.question = question
        # Its own identity, not the timestamp: two turns can begin inside the
        # same microsecond, and keying on `started` let one quietly overwrite the
        # other.
        self.id = uuid.uuid4().hex[:12]
        self.started = time.time()
        self.steps: list[dict] = []
        self.tools: list[dict] = []
        self.searches: list[dict] = []
        self.relaxed: list[str] = []
        self.knowledge: list[str] = []
        self.outfits: int = 0
        self.error: str | None = None

    def step(self, name: str, **fields: Any) -> None:
        self.steps.append({"name": name, "at": round(time.time() - self.started, 3), **fields})

    def tool(self, name: str, ok: bool, **fields: Any) -> None:
        self.tools.append({"tool": name, "ok": ok, **fields})

    def search(self, source: str, kept: int, raw: int, **fields: Any) -> None:
        self.searches.append(
            {"source": source, "kept": kept, "raw": raw, **fields}
        )

    def gave_up(self, reasons: list[str]) -> None:
        for reason in reasons:
            if reason not in self.relaxed:
                self.relaxed.append(reason)

    def used(self, formula_ids: list[str], trend_ids: list[str]) -> None:
        for value in [*(formula_ids or []), *(trend_ids or [])]:
            if value not in self.knowledge:
                self.knowledge.append(value)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "question": self.question,
            "started": self.started,
            "duration_ms": int((time.time() - self.started) * 1000),
            "steps": self.steps,
            "tools": self.tools,
            "searches": self.searches,
            "relaxed": self.relaxed,
            "knowledge": self.knowledge,
            "outfits": self.outfits,
            "error": self.error,
        }


def _ratio(kept: int, raw: int) -> float:
    return round(kept / raw, 3) if raw else 0.0


def _mean(values: list[int]) -> int:
    return int(sum(values) / len(values)) if values else 0


def search_quality(
    registry: Registry,
    seconds: float = WINDOW_SECONDS,
) -> dict:
    """Whether the search is finding things, and what it throws away."""
    events = registry._window("search", seconds)
    by_source: dict[str, dict] = {}

    for sample in events:
        name = str(sample.fields.get("source") or "unknown")
        bucket = by_source.setdefault(
            name,
            {
                "runs": 0,
                "failures": 0,
                "raw": 0,
                "kept": 0,
                "latency_ms": [],
                "relaxed": Counter(),
                "filtered": Counter(),
            },
        )
        bucket["runs"] += 1
        bucket["raw"] += int(sample.fields.get("raw") or 0)
        bucket["kept"] += int(sample.fields.get("kept") or 0)
        bucket["latency_ms"].append(int(sample.fields.get("latency_ms") or 0))

        if not sample.fields.get("ok"):
            bucket["failures"] += 1

        for reason in sample.fields.get("relaxed") or []:
            bucket["relaxed"][reason] += 1

        for reason, count in (sample.fields.get("filtered_out") or {}).items():
            bucket["filtered"][reason] += int(count)

    sources = {}

    for name, bucket in sorted(by_source.items()):
        sources[name] = {
            "runs": bucket["runs"],
            "failures": bucket["failures"],
            "failure_rate": round(bucket["failures"] / bucket["runs"], 3)
            if bucket["runs"]
            else 0.0,
            "kept": bucket["kept"],
            "raw": bucket["raw"],
            "keep_rate": _ratio(bucket["kept"], bucket["raw"]),
            "latency_ms": _mean(bucket["latency_ms"]),
            "relaxed": dict(bucket["relaxed"].most_common(5)),
            "dropped": dict(bucket["filtered"].most_common(5)),
        }

    return {
        "runs": len(events),
        "sources": sources,
        # The question a rate answers: are we finding fewer things, or throwing
        # away more?
        "questions": _search_questions(by_source),
    }


def _search_questions(by_source: dict[str, dict]) -> list[str]:
    notes = []

    for name, bucket in by_source.items():
        raw = bucket["raw"]

        if raw >= 20 and _ratio(bucket["kept"], raw) < 0.2:
            notes.append(
                f"{name}: из {raw} найденных остаётся {bucket['kept']} —"
                " фильтры режут почти всё"
            )

        for reason, count in bucket["filtered"].most_common(2):
            if raw and count / raw > 0.3:
                notes.append(f"{name}: «{reason}» отсекает больше трети результатов")

        for reason, count in bucket["relaxed"].most_common(1):
            if count:
                notes.append(f"{name}: «{reason}» приходилось ослаблять {count} раз")

    return notes[:8]


def tool_reliability(
    registry: Registry,
    seconds: float = WINDOW_SECONDS,
) -> dict:
    events = registry._window("tool", seconds)
    by_tool: dict[str, dict] = {}

    for sample in events:
        name = str(sample.fields.get("tool") or "unknown")
        bucket = by_tool.setdefault(name, {"runs": 0, "failures": 0, "latency_ms": []})
        bucket["runs"] += 1
        bucket["latency_ms"].append(int(sample.fields.get("latency_ms") or 0))

        if not sample.fields.get("ok"):
            bucket["failures"] += 1

    tools = {}

    for name, bucket in sorted(by_tool.items()):
        tools[name] = {
            "runs": bucket["runs"],
            "failures": bucket["failures"],
            "failure_rate": round(bucket["failures"] / bucket["runs"], 3)
            if bucket["runs"]
            else 0.0,
            "latency_ms": _mean(bucket["latency_ms"]),
        }

    return {"runs": len(events), "tools": tools}


def knowledge_coverage(
    registry: Registry,
    seconds: float = WINDOW_SECONDS,
) -> dict:
    """Are the knowledge cards being used, or are they decoration?

    A collection that is never drawn from is a collection nobody is paying for,
    and this is the only place that becomes visible.
    """
    events = registry._window("knowledge", seconds)
    used: Counter = Counter()
    turns = 0
    with_knowledge = 0
    items_covered = 0
    items_total = 0

    for sample in events:
        turns += 1
        formulas = sample.fields.get("formulas") or []
        trends = sample.fields.get("trends") or []

        for value in formulas:
            used[f"formula:{value}"] += 1

        for value in trends:
            used[f"trend:{value}"] += 1

        if formulas or trends:
            with_knowledge += 1

        items_covered += int(sample.fields.get("used_items") or 0)
        items_total += int(sample.fields.get("used_items") or 0) + int(
            sample.fields.get("bought_items") or 0
        )

    return {
        "turns": turns,
        "turns_with_knowledge": with_knowledge,
        "coverage": round(with_knowledge / turns, 3) if turns else 0.0,
        "distinct_used": len(used),
        "most_used": dict(used.most_common(5)),
        "items_from_wardrobe": items_covered,
        "items_total": items_total,
        "wardrobe_ratio": _ratio(items_covered, items_total),
    }


def turn_health(
    registry: Registry,
    seconds: float = WINDOW_SECONDS,
) -> dict:
    events = registry._window("turn", seconds)
    durations = [int(sample.fields.get("duration_ms") or 0) for sample in events]
    outfits = sum(int(sample.fields.get("outfits") or 0) for sample in events)
    errors = sum(1 for sample in events if sample.fields.get("had_error"))

    return {
        "turns": len(events),
        "errors": errors,
        "error_rate": round(errors / len(events), 3) if events else 0.0,
        "outfits": outfits,
        "outfits_per_turn": round(outfits / len(events), 2) if events else 0.0,
        "duration_ms": _mean(durations),
        "slowest_ms": max(durations) if durations else 0,
    }


_registry: Registry | None = None


def get_registry() -> Registry:
    global _registry

    if _registry is None:
        _registry = Registry()

    return _registry


def reset_registry() -> None:
    global _registry

    _registry = None


def report(
    registry: Registry | None = None,
    seconds: float = WINDOW_SECONDS,
) -> dict:
    """Every question in one place.

    The registry is an argument rather than a hidden global so that a caller
    reading a report is reading the same numbers it just recorded.
    """
    registry = registry if registry is not None else get_registry()

    return {
        "search": search_quality(registry, seconds),
        "tools": tool_reliability(registry, seconds),
        "knowledge": knowledge_coverage(registry, seconds),
        "turns": turn_health(registry, seconds),
    }


def report_ru(data: dict) -> list[str]:
    """The summary, in a language an operator can read at a glance."""
    lines = []

    search = data["search"]

    # A fresh process has nothing to report, and a page that always says "0
    # обращений" is a page people learn to stop reading.
    if not search["runs"]:
        return []

    lines.append(
        f"Поиск: {search['runs']} обращений, "
        f"{sum(s['kept'] for s in search['sources'].values())} вещей оставлено."
    )

    for note in search["questions"]:
        lines.append(f"  {note}")

    for name, bucket in search["sources"].items():
        if bucket["runs"] < 3:
            continue

        lines.append(
            f"  {name}: сработал в {round((1 - bucket['failure_rate']) * 100)}% случаев,"
            f" оставил {bucket['kept']} из {bucket['raw']},"
            f" {bucket['latency_ms']} мс"
        )

    tools = data["tools"]

    if tools["runs"]:
        for name, bucket in tools["tools"].items():
            lines.append(
                f"Инструмент {name}: {bucket['runs']} раз, отказов {bucket['failures']}"
            )

    knowledge = data["knowledge"]

    if knowledge["turns"]:
        lines.append(
            f"Знания: использованы в {round(knowledge['coverage'] * 100)}% образов,"
            f" карточек затронуто {knowledge['distinct_used']},"
            f" вещей из гардероба {knowledge['wardrobe_ratio'] * 100:.0f}%"
        )

    turns = data["turns"]

    if turns["turns"]:
        lines.append(
            f"Ходы: {turns['turns']}, ошибок {turns['errors']},"
            f" в среднем {turns['duration_ms']} мс,"
            f" образов за ход {turns['outfits_per_turn']}"
        )

    return lines
