def merge_products(
    current: list[dict] | None,
    update: list[dict] | None,
) -> list[dict]:
    products_by_id = {
        product["id"]: product
        for product in current or []
    }

    for product in update or []:
        existing = products_by_id.get(
            product["id"]
        )

        if not existing:
            products_by_id[
                product["id"]
            ] = product

            continue

        existing_attributes = set(
            existing.get(
                "search_desired_attributes",
                [],
            )
        )

        new_attributes = set(
            product.get(
                "search_desired_attributes",
                [],
            )
        )

        existing[
            "search_desired_attributes"
        ] = sorted(
            existing_attributes
            | new_attributes
        )

    return list(
        products_by_id.values()
    )