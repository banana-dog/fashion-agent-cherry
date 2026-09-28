import sqlite3
from datetime import UTC, datetime
from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from fashion_agent import storage
from fashion_agent.storage import (
    SQLiteStore,
    build_checkpointer,
    build_store,
    data_path,
    store_namespace,
)


def test_data_path_honours_env_and_creates_parent(monkeypatch, tmp_path):
    monkeypatch.setenv("CHERRY_STORE_DB", str(tmp_path / "nested" / "store.db"))

    assert data_path("CHERRY_STORE_DB", "default.db") == tmp_path / "nested" / "store.db"
    assert (tmp_path / "nested").is_dir()

    monkeypatch.delenv("CHERRY_STORE_DB")

    assert data_path("CHERRY_STORE_DB", "default.db").name == "default.db"


def test_data_path_resolves_relative_paths_against_project_root(monkeypatch):
    monkeypatch.setenv("CHERRY_STORE_DB", "data/relative.db")

    path = data_path("CHERRY_STORE_DB", "default.db")

    assert path == storage.PROJECT_ROOT / "data" / "relative.db"


def test_put_get_search_round_trip(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    namespace = store_namespace("alice")

    store.put(namespace, "color:black", {"category": "color", "target": "black"})

    item = store.get(namespace, "color:black")

    assert item is not None
    assert item.value["target"] == "black"
    assert item.namespace == namespace
    assert isinstance(item.created_at, datetime)


def test_missing_key_returns_none(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")

    assert store.get(store_namespace("alice"), "color:black") is None


def test_search_exact_namespace_excludes_nested_ones(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")

    store.put(("users", "alice", "style_preferences"), "color:black", {"a": 1})
    store.put(("users", "alice", "profile"), "sizes", {"a": 2})

    keys = [item.key for item in store.search(("users", "alice", "style_preferences"))]

    assert keys == ["color:black"]


def test_search_prefix_returns_nested_namespaces(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")

    store.put(("users", "alice", "style_preferences"), "color:black", {"a": 1})
    store.put(("users", "alice", "profile"), "sizes", {"a": 2})
    store.put(("users", "bob", "style_preferences"), "color:red", {"a": 3})

    found = {(item.namespace, item.key) for item in store.search(("users",))}

    assert found == {
        (("users", "alice", "style_preferences"), "color:black"),
        (("users", "alice", "profile"), "sizes"),
        (("users", "bob", "style_preferences"), "color:red"),
    }


def test_search_prefix_escapes_like_wildcards(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    wildcard = store_namespace("100%_off")

    store.put(wildcard, "color:black", {"a": 1})
    store.put(store_namespace("other"), "color:red", {"a": 2})

    found = [item.key for item in store.search(("users",))]
    scoped = [item.key for item in store.search(wildcard)]

    assert len(found) == 2
    assert scoped == ["color:black"]


def test_search_filter_and_pagination(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    namespace = store_namespace("alice")

    for index in range(5):
        store.put(namespace, f"item:{index}", {"category": "item", "index": index})

    filtered = store.search(namespace, filter={"category": "item", "index": 3})
    page = store.search(namespace, limit=2, offset=2)

    assert [item.key for item in filtered] == ["item:3"]
    assert [item.key for item in page] == ["item:2", "item:3"]


def test_delete_removes_entry(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    namespace = store_namespace("alice")

    store.put(namespace, "color:black", {"a": 1})
    store.delete(namespace, "color:black")

    assert store.search(namespace) == []


def test_put_updates_value_and_keeps_created_at(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    namespace = store_namespace("alice")

    store.put(namespace, "color:black", {"polarity": "like"})
    created_at = store.get(namespace, "color:black").created_at

    store.put(namespace, "color:black", {"polarity": "dislike"})
    item = store.get(namespace, "color:black")

    assert item.value["polarity"] == "dislike"
    assert item.created_at == created_at


def test_put_none_value_deletes_entry(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    namespace = store_namespace("alice")

    store.put(namespace, "color:black", {"a": 1})
    store.put(namespace, "color:black", None)

    assert store.search(namespace) == []


def test_data_survives_reopening_the_store(tmp_path):
    path = tmp_path / "store.db"
    namespace = store_namespace("alice")

    SQLiteStore(path).put(namespace, "fit:oversized", {"a": 1})

    reopened = SQLiteStore(path)

    assert [item.key for item in reopened.search(namespace)] == ["fit:oversized"]


def test_list_namespaces_prefix_depth_and_suffix(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")

    store.put(("users", "alice", "style_preferences"), "a", {"a": 1})
    store.put(("users", "bob"), "b", {"a": 1})
    store.put(("docs", "readme"), "c", {"a": 1})

    assert store.list_namespaces() == [
        ("docs", "readme"),
        ("users", "alice", "style_preferences"),
        ("users", "bob"),
    ]
    assert store.list_namespaces(prefix=("users",)) == [
        ("users", "alice", "style_preferences"),
        ("users", "bob"),
    ]
    assert store.list_namespaces(prefix=("users",), max_depth=2) == [("users", "bob")]


def test_concurrent_writes_from_threads(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    errors: list[BaseException] = []

    def write(index: int) -> None:
        try:
            store.put(store_namespace("alice"), f"color:c{index}", {"index": index})
        except Exception as error:  # noqa: BLE001 - any failure fails the test
            errors.append(error)

    threads = [__import__("threading").Thread(target=write, args=(i,)) for i in range(20)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert errors == []
    assert len(store.search(store_namespace("alice"), limit=50)) == 20


def test_abatch_matches_batch(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")
    namespace = store_namespace("alice")

    import asyncio

    from langgraph.store.base import PutOp, SearchOp

    ops = [
        PutOp(namespace, "color:black", {"a": 1}),
        SearchOp(namespace, None, 10, 0, None, None),
    ]

    results = asyncio.run(store.abatch(ops))

    assert results[0] is None
    assert [item.key for item in results[1]] == ["color:black"]


def test_unsupported_operation_raises(tmp_path):
    store = SQLiteStore(tmp_path / "store.db")

    with pytest.raises(NotImplementedError):
        store.batch([object()])


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _echo_graph(checkpointer):
    builder = StateGraph(_State)
    builder.add_node(
        "echo",
        lambda state: {
            "messages": [AIMessage(content=f"ok {state['messages'][-1].content}")]
        },
    )
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)

    return builder.compile(checkpointer=checkpointer)


def test_checkpointer_persists_history_across_processes(tmp_path, monkeypatch):
    db_path = tmp_path / "checkpoints.db"
    monkeypatch.setenv("CHERRY_CHECKPOINT_DB", str(db_path))

    monkeypatch.setattr(storage, "_CHECKPOINTER", None)
    config = {"configurable": {"thread_id": "thread-1"}}

    graph = _echo_graph(build_checkpointer())
    graph.invoke({"messages": [HumanMessage(content="first")]}, config)
    graph.invoke({"messages": [HumanMessage(content="second")]}, config)

    # A new process means a new saver instance over the same file.
    monkeypatch.setattr(storage, "_CHECKPOINTER", None)
    reopened = _echo_graph(build_checkpointer())

    history = [message.content for message in reopened.get_state(config).values["messages"]]

    assert history == ["first", "ok first", "second", "ok second"]


def test_build_checkpointer_is_process_wide(monkeypatch, tmp_path):
    monkeypatch.setenv("CHERRY_CHECKPOINT_DB", str(tmp_path / "checkpoints.db"))
    monkeypatch.setattr(storage, "_CHECKPOINTER", None)

    assert isinstance(build_checkpointer(), BaseCheckpointSaver)
    assert build_checkpointer() is build_checkpointer()


def test_build_store_writes_wal_and_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("CHERRY_STORE_DB", str(tmp_path / "store.db"))

    store = build_store()
    namespace = store_namespace("alice")
    store.put(namespace, "color:black", {"a": 1})

    connection = sqlite3.connect(tmp_path / "store.db")

    assert connection.execute(
        "SELECT journal_mode FROM pragma_journal_mode"
    ).fetchone()[0] == "wal"
    assert connection.execute("SELECT COUNT(*) FROM store_items").fetchone()[0] == 1
    assert connection.execute(
        "SELECT updated_at FROM store_items"
    ).fetchone()[0].endswith(tuple("+") + tuple(str(datetime.now(UTC).utcoffset())))
