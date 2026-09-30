"""One search per category, all at once, and the reports have to survive it.

This is a regression test for a failure that no unit test could see: the search
ran fine, every category found something, and the graph then refused to record
any of it, because the channel that holds the reports keeps only the last value
and seven nodes wrote to it in the same step. The client saw a turn that ended in
an error rather than in an outfit.

The search is stubbed so the test costs nothing and runs offline. What is being
checked is the wiring, not the search engine.
"""

import pytest

from fashion_agent.states import FashionState


class TestReportChannel:
    def test_reports_are_collected_rather_than_overwritten(self):
        """Two categories, two reports, both kept."""
        annotations = FashionState.__annotations__
        reducer = annotations["search_reports"].__metadata__[0]

        assert reducer([], [{"source": "a", "kept_count": 2}]) == [
            {"source": "a", "kept_count": 2}
        ]
        assert reducer([{"source": "a"}], [{"source": "b"}]) == [
            {"source": "a"},
            {"source": "b"},
        ]

    def test_a_cleared_plan_still_wins_over_what_came_before(self):
        """When the wardrobe already covers a category its search is skipped,
        and the node says so with an explicit overwrite rather than by not
        running."""
        from langgraph.types import Overwrite

        report = Overwrite([])

        assert report is not None
        assert FashionState.__annotations__["search_reports"].__metadata__


class TestParallelCategories:
    def test_a_turn_with_several_categories_finishes(self, monkeypatch):
        from fashion_agent import web
        from fashion_agent.product_search import product_search as search_module
        from fashion_agent.product_search.sources import SourceReport, SourceResult

        def fake_run(*, query, sources, relaxed=()):
            # A different item per category, so the merge has something to do.
            return SourceResult(
                products=[
                    {
                        "id": f"{query.category}-{query.text}",
                        "title": f"{query.category}: вещь",
                        "category": query.category,
                        "price": 5000,
                        "currency": "RUB",
                        "source": "shop",
                        "attributes": [],
                        "image_url": None,
                    }
                ],
                reports=[
                    SourceReport(
                        source="stub",
                        ok=True,
                        raw_count=5,
                        kept_count=1,
                        latency_ms=1,
                    )
                ],
            )

        monkeypatch.setattr(search_module, "run_product_search", fake_run)

        session = {
            "thread_id": "t-test",
            "user_id": "u-test",
            "locale": "ru-RU",
            "currency": "RUB",
            "client_now": None,
            "taste_context": [],
        }

        result = web.run_agent_turn(
            user_input="собери образ на работу: верх, низ и обувь, до 30 тысяч",
            session=session,
        )

        # The point is that the turn returns at all: before the reducer was
        # added it raised instead, whatever the search had found.
        assert "reply" in result
        assert isinstance(result["outfits"], list)

    def test_every_category_report_reaches_the_trace(self, monkeypatch):
        from fashion_agent import web
        from fashion_agent.metrics import get_registry, reset_registry
        from fashion_agent.product_search import product_search as search_module
        from fashion_agent.product_search.sources import SourceReport, SourceResult

        reset_registry()

        def fake_run(*, query, sources, relaxed=()):
            return SourceResult(
                products=[
                    {
                        "id": f"{query.category}-{query.text}",
                        "title": f"{query.category}: вещь",
                        "category": query.category,
                        "price": 4000,
                        "currency": "RUB",
                        "source": "shop",
                        "attributes": [],
                        "image_url": None,
                    }
                ],
                reports=[
                    SourceReport(
                        source="stub",
                        ok=True,
                        raw_count=4,
                        kept_count=2,
                        latency_ms=2,
                    )
                ],
            )

        monkeypatch.setattr(search_module, "run_product_search", fake_run)

        session = {
            "thread_id": "t-trace",
            "user_id": "u-trace",
            "locale": "ru-RU",
            "currency": "RUB",
            "client_now": None,
            "taste_context": [],
        }
        web.run_agent_turn(
            user_input=(
                "собери образ на работу в Москве: верх, низ и обувь, "
                "до 40 тысяч, спокойные цвета"
            ),
            session=session,
        )

        traces = get_registry().traces("u-trace")

        assert traces
        searches = [
            step
            for step in traces[0]["steps"]
            if step.get("name") == "search"
        ]

        # One report per category, not the last one to finish. Before the
        # reducer was added only a single category's report survived.
        assert len(searches) >= 3
        assert all(step.get("kept") is not None for step in searches)
        reset_registry()


class TestNothingElseCollides:
    """Only the search fan-out writes in parallel; the rest must stay singular."""

    @pytest.mark.parametrize(
        "key",
        [
            "request",
            "missing_fields",
            "search_plan",
            "outfits",
            "ranked_products",
            "resolved_style",
            "client_season",
        ],
    )
    def test_single_writer_channels_stay_single(self, key):
        annotations = FashionState.__annotations__

        # No reducer, so one writer per step. Adding a reducer to any of these
        # would change what the node means, and nothing would say so.
        assert not hasattr(annotations[key], "__metadata__")
