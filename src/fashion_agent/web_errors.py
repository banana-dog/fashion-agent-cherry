"""What the client is told when something underneath breaks.

A turn that dies takes the conversation with it, and a client who sees a raw
Python error learns nothing about what to do next. So failures are described in
the client's own language, by kind rather than by text: what went wrong, and
whether trying again is worth it.

What is deliberately not here is the provider's own words. An upstream message
can carry an account name, a request body, an internal host, or a stack of
framework internals; none of that belongs in a chat window. The type is kept
because it is useful to whoever reads the log, and dropped everywhere a client
can see it.

Which failures deserve a retry is stated rather than guessed. A 429 from a model
is worth another attempt in a moment; a malformed answer is not going to improve
on its own; a database that is locked will probably still be locked in a second.
"""

from typing import Any

# Kinds a client may act on, mapped to what we say and whether to retry.
_CURATED: list[tuple[type, str, bool]] = []


def _curated(*types: type, message: str, retry: bool) -> None:
    for kind in types:
        _CURATED.append((kind, message, retry))


def _lazy() -> None:
    """Built on first use so importing this module stays cheap.

    The type list covers the libraries a failure can come out of, and importing
    them at module load would make every request path pay for it.
    """
    if _CURATED:
        return

    import httpx
    import pydantic

    _curated(
        httpx.TimeoutException,
        message="Источник не ответил вовремя. Попробуйте ещё раз.",
        retry=True,
    )
    _curated(
        httpx.TransportError,
        message="Не удалось связаться с источником. Проверьте соединение.",
        retry=True,
    )
    _curated(
        pydantic.ValidationError,
        message="Ответ пришёл в неожиданном виде, и я не смогла его разобрать.",
        retry=False,
    )


# Import errors are matched by name so this module does not need the provider's
# package to be installed at all.
_BY_NAME: dict[str, tuple[str, bool]] = {
    "APIConnectionError": ("Не удалось связаться с моделью.", True),
    "APITimeoutError": ("Модель не ответила вовремя.", True),
    "RateLimitError": ("Слишком много запросов подряд. Немного подожду.", True),
    "InternalServerError": ("У модели внутренняя ошибка. Попробуйте ещё раз.", True),
    "APIStatusError": ("Модель отклонила запрос.", False),
    "AuthenticationError": ("Ключ к модели отвергнут. Проверьте настройки.", False),
    "ToolUnavailable": ("Инструмент недоступен.", False),
    "OutputParserException": ("Ответ модели не удалось разобрать.", False),
    "OperationalError": ("База занята другим запросом. Попробуйте ещё раз.", True),
    "IntegrityError": ("Не удалось записать данные.", False),
    "JSONDecodeError": ("Источник вернул не тот ответ, который я ждала.", False),
    "KeyError": ("В ответе не хватило нужного поля.", False),
}


def describe(error: BaseException) -> str:
    """A sentence for a client, and nothing they did not need to know."""
    _lazy()

    for kind, message, _retry in _CURATED:
        if isinstance(error, kind):
            return message

    for klass in type(error).__mro__:
        found = _BY_NAME.get(klass.__name__)

        if found is not None:
            return found[0]

    if isinstance(error, PermissionError):
        return "Нет доступа к данным. Попробуйте ещё раз."

    if isinstance(error, OSError):
        return "Не удалось прочитать или записать данные. Попробуйте ещё раз."

    if isinstance(error, (ValueError, TypeError, RuntimeError, KeyError)):
        return "Что-то пошло не так на моей стороне. Попробуйте ещё раз."

    return "Что-то пошло не так на моей стороне."


def worth_retrying(error: BaseException) -> bool:
    """Whether another attempt in a moment is likely to work."""
    _lazy()

    for kind, _message, retry in _CURATED:
        if isinstance(error, kind):
            return retry

    for klass in type(error).__mro__:
        found = _BY_NAME.get(klass.__name__)

        if found is not None:
            return found[1]

    return isinstance(error, (TimeoutError, ConnectionError))


def detail(error: BaseException) -> dict[str, Any]:
    """What a log gets, which may name the class but never the text.

    The message of an upstream exception is the one thing here that could carry
    somebody else's data, so it is reduced to the class name and the curated
    sentence.
    """
    return {
        "kind": type(error).__name__,
        "message": describe(error),
        "retryable": worth_retrying(error),
    }


def safe_detail(error: BaseException) -> str:
    """One line for a log file, in the operator's language."""
    return f"{type(error).__name__}: {describe(error)}"
