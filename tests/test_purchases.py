"""Purchase history, which only exists because the client wrote it down.

Nothing here may be inferred. A price in a search result is a price at the
moment of searching; a price in a shop is a different number, and the difference
is money. So every figure in this file comes from a sentence the client typed.
"""

import pytest

from fashion_agent.purchases import (
    Purchase,
    PurchaseStore,
    recent_months,
    summarise,
)

PHRASE = "parol-dlinnaya-1"


@pytest.fixture
def store(tmp_path):
    return PurchaseStore(tmp_path / "purchases.sqlite3")


class TestWhatCountsAsAPurchase:
    def test_a_title_is_needed(self):
        with pytest.raises(ValueError):
            Purchase(title="   ")

    def test_extra_spaces_in_a_title_are_tidy(self):
        assert Purchase(title="  Тренч   чёрный ").title == "Тренч чёрный"

    def test_a_price_is_optional(self):
        assert Purchase(title="Джемпер").paid is None

    def test_a_negative_price_is_refused(self):
        with pytest.raises(ValueError):
            Purchase(title="Джемпер", paid=-1)

    def test_an_absurd_price_is_refused(self):
        with pytest.raises(ValueError):
            Purchase(title="Джемпер", paid=99_000_000)

    def test_a_something_that_is_not_a_link_is_not_stored_as_one(self):
        """A made-up scheme would produce a citation that goes nowhere."""
        assert Purchase(title="Джемпер", url="javascript:alert(1)").url is None
        assert Purchase(title="Джемпер", url="lamoda.ru/p/1").url is None

    def test_a_real_link_is_kept(self):
        assert (
            Purchase(title="Джемпер", url="https://www.lamoda.ru/p/1").url
            == "https://www.lamoda.ru/p/1"
        )

    def test_a_long_title_is_truncated_rather_than_rejected(self):
        assert len(Purchase(title="я" * 500).title) == 120


class TestStore:
    def test_a_purchase_is_kept(self, store):
        store.add("u1", title="Тренч", paid=18900)

        [item] = store.purchases("u1")

        assert item.title == "Тренч"
        assert item.paid == 18900

    def test_newest_first(self, store):
        first = store.add("u1", title="Тренч")
        second = store.add("u1", title="Джемпер")

        assert [item.id for item in store.purchases("u1")] == [second.id, first.id]

    def test_one_client_does_not_see_another(self, store):
        store.add("u1", title="Тренч")
        store.add("u2", title="Пальто")

        assert [item.title for item in store.purchases("u1")] == ["Тренч"]

    def test_deleting_someone_elses_purchase_is_refused(self, store):
        theirs = store.add("u2", title="Пальто")

        assert store.delete("u1", theirs.id) is False
        assert len(store.purchases("u2")) == 1

    def test_forgetting_removes_only_that_person(self, store):
        store.add("u1", title="Тренч")
        store.add("u2", title="Пальто")

        assert store.forget("u1") == 1
        assert store.purchases("u1") == []
        assert len(store.purchases("u2")) == 1

    def test_a_purchase_survives_a_reopen(self, tmp_path):
        path = tmp_path / "purchases.sqlite3"
        PurchaseStore(path).add("u1", title="Тренч")

        assert len(PurchaseStore(path).purchases("u1")) == 1

    def test_a_purchase_can_point_at_a_wardrobe_item(self, store):
        item = store.add("u1", title="Тренч", wardrobe_item_id="w1")

        assert store.purchases("u1")[0].wardrobe_item_id == "w1"
        assert item.wardrobe_item_id == "w1"


class TestSummary:
    def test_the_total_is_the_sum_of_what_she_said(self, store):
        store.add("u1", title="Тренч", paid=18900)
        store.add("u1", title="Джемпер", paid=3100)

        assert summarise(store.purchases("u1")).total == 22000

    def test_a_purchase_without_a_price_is_counted_but_not_summed(self, store):
        store.add("u1", title="Тренч", paid=18900)
        store.add("u1", title="Джемпер")

        report = summarise(store.purchases("u1"))

        assert report.total == 18900
        assert report.without_price == 1
        assert report.count == 2

    def test_an_empty_history_totals_nothing(self, store):
        report = summarise([])

        assert report.count == 0
        assert report.total == 0
        assert report.by_month == []

    def test_months_are_grouped_and_newest_first(self, store):
        store.add("u1", title="Тренч", paid=100)
        store.add("u1", title="Джемпер", paid=50)

        report = summarise(store.purchases("u1"))

        assert report.by_month[0]["total"] == 150
        assert report.by_month[0]["month"] in recent_months()

    def test_a_month_is_read_off_the_record(self):
        assert Purchase(title="x", created_at="2026-03-04T00:00:00Z").month == "2026-03"


@pytest.fixture
def cabinet(tmp_path, monkeypatch):
    """A signed-in client with an isolated set of databases."""
    monkeypatch.setenv("CHERRY_WARDROBE_DB", str(tmp_path / "wardrobe.sqlite3"))
    monkeypatch.setenv("CHERRY_WARDROBE_IMAGES", str(tmp_path / "images"))
    monkeypatch.setenv("CHERRY_LOOKS_DB", str(tmp_path / "looks.sqlite3"))
    monkeypatch.setenv("CHERRY_TASTE_DB", str(tmp_path / "taste.sqlite3"))
    monkeypatch.setenv("CHERRY_ACCOUNTS_DB", str(tmp_path / "accounts.sqlite3"))
    monkeypatch.setenv("CHERRY_STORE_DB", str(tmp_path / "store.sqlite3"))

    from fashion_agent.accounts import Accounts
    from fashion_agent.cabinet import build
    from fashion_agent.cabinet_web import render
    from fashion_agent.look_session import reset_look_store
    from fashion_agent.purchases import get_purchase_store, reset_purchase_store
    from fashion_agent.wardrobe import reset_wardrobe

    reset_wardrobe()
    reset_look_store()
    reset_purchase_store()

    accounts = Accounts()
    accounts.register("mira", PHRASE)
    user_id = accounts.sign_in("mira", PHRASE).user_id

    yield {
        "user_id": user_id,
        "build": build,
        "render": render,
        "purchases": get_purchase_store(),
        "accounts": accounts,
    }

    reset_purchase_store()
    reset_look_store()
    reset_wardrobe()


class TestInTheCabinet:
    def test_an_empty_history_still_admits_itself(self, cabinet):
        view = cabinet["build"](cabinet["user_id"])

        assert view.purchases["known"] is False
        assert "не знаю" in view.purchases["reason"]

    def test_a_written_purchase_fills_the_section(self, cabinet):
        cabinet["purchases"].add(cabinet["user_id"], title="Тренч", paid=18900)
        view = cabinet["build"](cabinet["user_id"])

        assert view.purchases["known"] is True
        assert view.purchases["total"] == 18900
        assert view.purchases["items"][0]["title"] == "Тренч"

    def test_the_page_shows_the_written_total(self, cabinet):
        cabinet["purchases"].add(cabinet["user_id"], title="Тренч", paid=18900)
        page = cabinet["render"](cabinet["build"](cabinet["user_id"]))

        assert "Тренч" in page
        # Written the way a person reads money, not the way an API returns it.
        assert "18\u2009900" in page and "\u20bd" in page
        # Scoped to the purchase section: the gaps section above it has its own
        # honest "I don't know" lines, which must not confuse the check.
        section = page.split("История покупок", 1)[1].split("</section>", 1)[0]
        assert "не знаю" not in section

    def test_a_purchase_without_a_price_is_shown_as_such(self, cabinet):
        cabinet["purchases"].add(cabinet["user_id"], title="Джемпер")
        page = cabinet["render"](cabinet["build"](cabinet["user_id"]))

        assert "без цены" in page

    def test_the_page_offers_a_way_to_record_one(self, cabinet):
        page = cabinet["render"](cabinet["build"](cabinet["user_id"]))

        assert 'id="buyForm"' in page

    def test_another_person_buying_does_not_show_up(self, cabinet):
        other = cabinet["accounts"].register("anton", PHRASE).user_id
        cabinet["purchases"].add(other, title="Пальто", paid=50000)
        view = cabinet["build"](cabinet["user_id"])

        assert view.purchases["known"] is False

    def test_purchases_leave_the_export(self, cabinet):
        from fashion_agent.privacy import collect, forget

        user_id = cabinet["user_id"]
        cabinet["purchases"].add(user_id, title="Тренч", paid=18900)
        other = cabinet["accounts"].register("anton", PHRASE).user_id
        cabinet["purchases"].add(other, title="Пальто", paid=50000)

        assert collect(user_id).counts["purchases"] == 1

        report = forget(user_id)

        assert report.removed["purchases"] == 1
        assert cabinet["purchases"].purchases(other)[0].title == "Пальто"
