import pytest

from fashion_agent.client_profile import ClientProfile, ClientSizes
from fashion_agent.llm import Context
from fashion_agent.product_search.models import ProductSearch, SearchPlan
from fashion_agent.product_search.product_search import (
    build_product_query,
    create_search_plan,
    dispatch_product_searches,
    run_product_search,
    search_one_category,
)
from fashion_agent.product_search.sources import (
    Marketplace,
    ProductQuery,
    ProductSource,
    SourceReport,
    SourceResult,
    SourceUnavailable,
)

WB_URL = "https://www.wildberries.ru/catalog/123456789/detail.aspx"


def product(price=3000, **overrides) -> dict:
    payload = {
        "id": "wildberries:123456789",
        "title": "Свитер кремовый",
        "category": "top",
        "price": price,
        "currency": "RUB",
        "sizes": [],
        "attributes": ["color:cream"],
        "source": "Wildberries",
        "marketplace": "wildberries",
        "url": WB_URL,
        "image_url": None,
        "rating": 4.5,
        "reviews": 10,
        "snippet": None,
        "position": 1,
    }

    payload.update(overrides)

    return payload


class FakeSource(ProductSource):
    name = "fake"

    def __init__(self, results, *, name=None):
        self.results = results
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.available_flag = True

        if name:
            self.name = name

    def available(self) -> bool:
        return self.available_flag

    def search(self, query, *, relaxed=()):
        self.calls.append((query.text, tuple(relaxed)))

        result = self.results.get((query.text, tuple(relaxed)))

        if result is not None:
            return result

        return SourceResult(
            reports=[SourceReport(source=self.name, ok=True, kept_count=0)]
        )


class StubExtractor:
    def __init__(self, value):
        self.value = value
        self.prompts: list = []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        return self.value


class Runtime:
    context = Context(user_id="alice")


@pytest.fixture
def runtime():
    return Runtime()


def test_build_product_query_uses_client_size():
    profile = ClientProfile(sizes=ClientSizes(top="M", shoe=38))

    query = build_product_query(
        {
            "category": "top",
            "query": "свитер",
            "keywords": ["шерсть"],
            "colors": ["color:cream"],
            "max_price": 5000,
        },
        location="Москва",
        client_profile=profile,
        locale="ru-RU",
        currency="RUB",
    )

    assert query.size == "M"
    assert query.price_max == 5000
    assert query.location == "Москва"
    assert query.keywords == ["шерсть"]


def test_build_product_query_uses_shoe_size_for_shoes():
    profile = ClientProfile(sizes=ClientSizes(shoe=38.5))

    query = build_product_query(
        {"category": "shoes", "query": "ботинки", "desired_attributes": []},
        location=None,
        client_profile=profile,
        locale="ru-RU",
        currency="RUB",
    )

    assert query.size == 38.5


def test_build_product_query_without_profile_size():
    query = build_product_query(
        {"category": "bag", "query": "сумка", "desired_attributes": []},
        location=None,
        client_profile=ClientProfile(),
        locale="ru-RU",
        currency="RUB",
    )

    assert query.size is None


def test_search_plan_field_defaults():
    search = ProductSearch(category="top", query="свитер", fallback_query="свитер")

    assert search.keywords == []
    assert search.colors == []
    assert search.brand is None
    assert search.price_min is None
    assert search.max_price is None


def test_run_product_search_stops_at_the_first_source_with_results():
    first = FakeSource({("свитер", ()): SourceResult(products=[product()])})
    second = FakeSource({("свитер", ()): SourceResult(products=[product()])})

    result = run_product_search(
        query=ProductQuery(category="top", text="свитер"),
        sources=[first, second],
    )

    assert len(result.products) == 1
    assert second.calls == []


def test_run_product_search_falls_through_to_the_next_source():
    first = FakeSource({}, name="shopping")
    second = FakeSource({("свитер", ()): SourceResult(products=[product()])})

    result = run_product_search(
        query=ProductQuery(category="top", text="свитер"),
        sources=[first, second],
    )

    assert result.products
    assert second.calls == [("свитер", ())]


def test_run_product_search_reports_unavailable_sources():
    first = FakeSource({})
    first.available_flag = False
    second = FakeSource({})

    result = run_product_search(
        query=ProductQuery(category="top", text="свитер"),
        sources=[first, second],
    )

    assert result.reports[0].error == "source is not configured"


def test_run_product_search_turns_unavailable_into_a_report():
    class Raising(FakeSource):
        def search(self, query, *, relaxed=()):
            raise SourceUnavailable("shop is down")

    result = run_product_search(
        query=ProductQuery(category="top", text="свитер"),
        sources=[Raising({})],
    )

    assert "shop is down" in result.errors[0]


def test_search_one_category_relaxes_a_suggested_constraint(runtime, monkeypatch):
    source = FakeSource(
        {
            ("свитер", ()): SourceResult(
                reports=[
                    SourceReport(
                        source="fake",
                        ok=True,
                        raw_count=40,
                        kept_count=0,
                        filtered_out={"price": 40},
                    )
                ],
                suggested_relaxations=["price"],
            ),
            ("свитер", ("price",)): SourceResult(
                products=[product(price=9000)],
                reports=[
                    SourceReport(
                        source="fake",
                        ok=True,
                        kept_count=1,
                        relaxed=["цена"],
                    )
                ],
            ),
        }
    )

    monkeypatch.setattr(
        "fashion_agent.product_search.product_search.get_sources",
        lambda: [source],
    )

    result = search_one_category(
        {
            "search": {
                "category": "top",
                "query": "свитер",
                "fallback_query": "свитер широкий",
                "desired_attributes": ["color:cream"],
                "max_price": 5000,
            },
            "location": "Москва",
            "client_profile": {},
        },
        runtime,
    )

    assert [item["price"] for item in result["products"]] == [9000]
    assert result["products"][0]["search_query"] == "свитер"
    assert [call[1] for call in source.calls] == [(), ("price",)]


def test_search_one_category_uses_the_fallback_query(runtime, monkeypatch):
    source = FakeSource(
        {
            ("свитер широкий", ()): SourceResult(products=[product()]),
        }
    )

    monkeypatch.setattr(
        "fashion_agent.product_search.product_search.get_sources",
        lambda: [source],
    )

    result = search_one_category(
        {
            "search": {
                "category": "top",
                "query": "свитер кремовый шерстяной",
                "fallback_query": "свитер широкий",
                "desired_attributes": [],
                "max_price": None,
            },
            "location": None,
            "client_profile": {},
        },
        runtime,
    )

    assert result["products"]
    assert result["products"][0]["search_query"] == "свитер широкий"


def test_search_one_category_reports_when_nothing_is_configured(runtime, monkeypatch):
    source = FakeSource({})
    source.available_flag = False

    monkeypatch.setattr(
        "fashion_agent.product_search.product_search.get_sources",
        lambda: [source],
    )

    result = search_one_category(
        {
            "search": {
                "category": "top",
                "query": "свитер",
                "fallback_query": "свитер",
                "desired_attributes": [],
                "max_price": None,
            },
            "location": None,
            "client_profile": {},
        },
        runtime,
    )

    assert result["products"] == []
    assert result["search_reports"][0]["ok"] is False


def test_dispatch_passes_the_client_profile():
    sends = dispatch_product_searches(
        {
            "request": {"location": "Казань"},
            "client_profile": {"sizes": {"top": "L"}},
            "search_plan": [
                {
                    "category": "top",
                    "query": "свитер",
                    "fallback_query": "свитер",
                    "desired_attributes": [],
                    "max_price": None,
                }
            ],
        }
    )

    assert len(sends) == 1
    assert sends[0].arg["location"] == "Казань"
    assert sends[0].arg["client_profile"]["sizes"]["top"] == "L"


def test_dispatch_survives_a_missing_profile():
    sends = dispatch_product_searches(
        {
            "request": {"location": None},
            "search_plan": [
                {
                    "category": "top",
                    "query": "свитер",
                    "fallback_query": "свитер",
                    "desired_attributes": [],
                    "max_price": None,
                }
            ],
        }
    )

    assert sends[0].arg["client_profile"] == ClientProfile().model_dump(mode="json")


def test_search_plan_drops_unknown_formula_and_trend_ids(monkeypatch):
    search = ProductSearch(
        category="top",
        query="свитер",
        fallback_query="свитер",
        desired_attributes=["color:cream"],
        max_price=100,
    )
    search.formula_ids = ["formula:known", "formula:invented"]
    search.trend_ids = ["trend:known", "trend:invented"]

    monkeypatch.setattr(
        "fashion_agent.product_search.product_search.search_plan_extractor",
        StubExtractor(SearchPlan(searches=[search])),
    )

    result = create_search_plan(
        {
            "request": {"occasion": "work"},
            "style_preferences": [],
            "client_profile": {},
            "resolved_style": None,
            "retrieved_outfit_formulas": [{"id": "formula:known"}],
            "retrieved_trends": [{"id": "trend:known"}],
        },
        Runtime(),
    )

    assert result["search_plan"][0]["formula_ids"] == ["formula:known"]
    assert result["search_plan"][0]["trend_ids"] == ["trend:known"]


def test_search_plan_resets_products_and_reports(monkeypatch):
    search = ProductSearch(
        category="top",
        query="свитер",
        fallback_query="свитер",
    )

    monkeypatch.setattr(
        "fashion_agent.product_search.product_search.search_plan_extractor",
        StubExtractor(SearchPlan(searches=[search])),
    )

    result = create_search_plan(
        {
            "request": {},
            "style_preferences": [],
            "client_profile": {},
            "resolved_style": None,
            "retrieved_outfit_formulas": [],
            "retrieved_trends": [],
        },
        Runtime(),
    )

    # A second pass in the same thread must not reuse the previous results.
    assert getattr(result["products"], "value", result["products"]) == []
    assert getattr(result["search_reports"], "value", result["search_reports"]) == []


def test_marketplace_any_is_available():
    assert Marketplace.ANY in list(Marketplace)
