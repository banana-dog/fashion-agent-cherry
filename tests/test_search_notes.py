from fashion_agent.outfits.presentation import (
    failure_messages,
    search_relaxation_notes,
)


def test_no_notes_without_reports():
    assert search_relaxation_notes({}) == []
    assert search_relaxation_notes({"search_reports": []}) == []


def test_collects_every_relaxed_constraint():
    notes = search_relaxation_notes(
        {
            "search_reports": [
                {"source": "serpapi-shopping", "relaxed": ["цена"]},
                {"source": "serpapi-web", "relaxed": []},
                {"source": "serpapi-web", "relaxed": ["размер"]},
            ]
        }
    )

    assert notes == ["цена", "размер"]


def test_deduplicates_a_repeated_relaxation():
    notes = search_relaxation_notes(
        {
            "search_reports": [
                {"source": "a", "relaxed": ["цена"]},
                {"source": "b", "relaxed": ["цена"]},
            ]
        }
    )

    assert notes == ["цена"]


def test_failed_reports_without_a_relaxation_stay_quiet():
    notes = search_relaxation_notes(
        {
            "search_reports": [
                {"source": "serpapi-shopping", "ok": False, "error": "timeout", "relaxed": []},
            ]
        }
    )

    assert notes == []


def test_failure_messages_cover_the_budget_case():
    messages = failure_messages(
        {
            "failure_type": "budget_too_low",
            "budget_max": 5000,
            "currency": "RUB",
            "cheapest_required_total": 9000,
            "budget_shortfall": 4000,
            "required_categories": ["top", "shoes"],
            "missing_categories": [],
            "relaxations": [],
        }
    )

    assert messages
    assert any("4 000" in message for message in messages)
