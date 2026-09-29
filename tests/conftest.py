import os
import sys
from pathlib import Path

for key in (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
):
    os.environ.pop(
        key,
        None,
    )


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# Every application database is redirected into the test's own directory.
# Without this, a test reads whatever the developer happened to collect, and a
# green suite means nothing: one knowledge assertion failed here because freshly
# gathered trend cards were still current in January 2027.
DATABASE_ENV = (
    "CHERRY_TASTE_DB",
    "CHERRY_STORE_DB",
    "CHERRY_CHECKPOINT_DB",
    "CHERRY_WARDROBE_DB",
    "CHERRY_WARDROBE_IMAGES",
    "CHERRY_TREND_DB",
    "CHERRY_SEARCH_CACHE",
)


def _clear_singletons():
    """Drop module-level caches so the new paths are picked up."""
    from fashion_agent.accounts import reset_accounts
    from fashion_agent.knowledge.repository import get_knowledge_repository
    from fashion_agent.look_session import reset_look_store
    from fashion_agent.storage import reset_checkpointer
    from fashion_agent.styleDNA import taste_quiz
    from fashion_agent.trends.scheduler import reset_scheduler
    from fashion_agent.wardrobe import reset_wardrobe

    reset_accounts()
    reset_checkpointer()
    reset_wardrobe()
    reset_look_store()
    reset_scheduler()
    taste_quiz.cache_clear()
    get_knowledge_repository.cache_clear()

    from fashion_agent import vision, web
    from fashion_agent.product_search import registry
    from fashion_agent.trends import store as trend_store

    registry.reset_sources()
    trend_store.reset_trend_store()
    vision.reset_vision_client()

    # The chat layer keeps its own conversation; it must not outlive the test.
    web._body_profile = None


def pytest_runtest_setup():
    import tempfile

    for key in DATABASE_ENV:
        os.environ.pop(key, None)

    # A fresh directory per test keeps the files apart as well as the settings.
    data_dir = Path(tempfile.mkdtemp(prefix="cherry-test-"))

    os.environ["CHERRY_TEST_DATA"] = str(data_dir)
    os.environ["CHERRY_TASTE_DB"] = str(data_dir / "taste.sqlite3")
    os.environ["CHERRY_STORE_DB"] = str(data_dir / "store.sqlite3")
    os.environ["CHERRY_CHECKPOINT_DB"] = str(data_dir / "checkpoints.sqlite3")
    os.environ["CHERRY_WARDROBE_DB"] = str(data_dir / "wardrobe.sqlite3")
    os.environ["CHERRY_WARDROBE_IMAGES"] = str(data_dir / "wardrobe_images")
    os.environ["CHERRY_LOOKS_DB"] = str(data_dir / "looks.sqlite3")
    os.environ["CHERRY_ACCOUNTS_DB"] = str(data_dir / "accounts.sqlite3")
    os.environ["CHERRY_TREND_DB"] = str(data_dir / "trends.sqlite3")
    os.environ["CHERRY_SEARCH_CACHE"] = str(data_dir / "search_cache.json")

    _clear_singletons()


def client_user_id(client) -> str:
    """Who the server thinks this browser is.

    Tests used to name a user themselves, which was possible only while every
    browser shared one name. Now the server decides, so a test that puts data in
    the wardrobe has to ask who it is being.
    """
    status, payload = client.request("GET", "/api/account")

    assert status == 200, payload

    return payload["user_id"]
