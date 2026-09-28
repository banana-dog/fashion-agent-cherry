"""Measure how relevant product search actually is.

Run it with:

    uv run python scripts/search_quality.py

It costs one SerpApi credit per case, so the case list is short and fixed. The
output is the number to watch when changing queries, filters or the source: the
share of returned products that are a real product page, in the client's size,
inside the budget, and matching the requested colour.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / "src" / "fashion_agent" / ".env")

from fashion_agent.product_search.registry import build_sources
from fashion_agent.product_search.sources import (
    Marketplace,
    ProductQuery,
)


@dataclass
class Case:
    name: str
    query: ProductQuery
    expect: dict[str, str] = field(default_factory=dict)
    categories: set[str] = field(default_factory=set)


def build_cases() -> list[Case]:
    return [
        Case(
            name="кремовый свитер, бюджет и размер",
            query=ProductQuery(
                category="top",
                text="кремовый свитер женский",
                colors=["color:cream"],
                size="44",
                price_max=5000,
                marketplaces=[Marketplace.WILDBERRIES],
            ),
            expect={"item": "свитер", "color": "cream"},
        ),
        Case(
            name="чёрные брюки оверсайз",
            query=ProductQuery(
                category="bottom",
                text="чёрные широкие брюки оверсайз",
                colors=["color:black"],
                size="42",
                price_max=6000,
                marketplaces=[Marketplace.WILDBERRIES],
            ),
            expect={"item": "брюк"},
        ),
        Case(
            name="ботинки на каблуке, без размера",
            query=ProductQuery(
                category="shoes",
                text="чёрные ботинки на каблуке",
                colors=["color:black"],
                price_max=15000,
                marketplaces=[Marketplace.WILDBERRIES, Marketplace.OZON],
            ),
            expect={"item": "ботин"},
        ),
        Case(
            name="кожаная сумка, другой бюджет",
            query=ProductQuery(
                category="bag",
                text="кожаная сумка структурированная",
                price_max=25000,
                marketplaces=[Marketplace.WILDBERRIES],
            ),
            expect={"item": "сумк"},
        ),
        Case(
            name="повседневное платье, минимализм",
            query=ProductQuery(
                category="dress",
                text="минималистичное платье миди",
                price_max=9000,
                marketplaces=[Marketplace.LAMODA, Marketplace.WILDBERRIES],
            ),
            expect={"item": "плать"},
        ),
    ]


def is_relevant(product: dict, case: Case) -> bool:
    from fashion_agent.product_search.snippets import matches_size

    title = product["title"].lower()

    if case.expect.get("item") and case.expect["item"] not in title:
        return False

    if (
        case.expect.get("color")
        and f"color:{case.expect['color']}" not in product["attributes"]
    ):
        return False

    if not matches_size(product.get("sizes", []), case.query.size):
        return False

    price = product.get("price")
    cap = case.query.price_max

    return not (price is not None and cap is not None and price > cap)


def run(case: Case, limit: int) -> dict:
    sources = build_sources()

    if not any(source.available() for source in sources):
        return {"case": case.name, "error": "no configured source"}

    products: list[dict] = []
    reports: list[dict] = []

    for source in sources:
        if not source.available():
            continue

        result = source.search(case.query.model_copy(update={"limit": limit}))
        products.extend(result.products)
        reports.extend(report.model_dump() for report in result.reports)

        if result.products:
            break

    relevant = [product for product in products if is_relevant(product, case)]

    return {
        "case": case.name,
        "query": case.query.text,
        "found": len(products),
        "relevant": len(relevant),
        "share": round(len(relevant) / len(products), 2) if products else 0.0,
        "prices": sorted(
            {
                product["price"]
                for product in products
                if product.get("price") is not None
            }
        )[:8],
        "sources": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--case", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cases = build_cases()

    if args.case:
        cases = [case for case in cases if args.case in case.name]

    results = [run(case, args.limit) for case in cases]

    for result in results:
        if "error" in result:
            print(f"SKIP  {result['case']}: {result['error']}")
            continue

        print(
            f"{result['relevant']}/{result['found']} "
            f"({result['share']:.0%})  {result['case']}"
        )
        print(f"      query: {result['query']}")
        print(f"      prices: {result['prices']}")

        for report in result["sources"]:
            status = "ok" if report["ok"] else "FAIL"

            print(
                f"      {status} {report['source']}: "
                f"raw={report['raw_count']} kept={report['kept_count']} "
                f"filtered={report['filtered_out']} "
                f"relaxed={report['relaxed']} error={report['error']}"
            )

            if args.verbose and report.get("search_text"):
                print(f"      sent: {report['search_text']}")

    scored = [result for result in results if "share" in result]

    if not scored:
        return 0

    total_share = sum(result["share"] for result in scored) / len(scored)

    print()
    print(f"cases: {len(scored)}  mean relevance: {total_share:.0%}")
    print(json.dumps({"mean_relevance": total_share}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
