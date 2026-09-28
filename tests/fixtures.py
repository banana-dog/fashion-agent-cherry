"""Fixtures captured from real search results on Russian marketplaces.

These snippets are the reason the strict parsers exist: every string here
produced a wrong price or a wrong size with the generous parsers.
"""

# Yandex, "site:wildberries.ru чёрные ботинки на каблуке"
BOOTS_TAG_SNIPPET = (
    "Ботинки демисезонные черные на каблуке лаковые SHOES OCEAN. ... "
    "Зимние ботинки берцы на устойчивом каблуке на молнии Naked Diamond."
    "Рейтинг 4,2 из 54,2"
)
BOOTS_DETAIL_SNIPPET = (
    "ботинки черные; ботинки на каблуке женские; ботильоны на высоком каблуке. "
    "Уход за обувью…\u2060В ПВЗ Курьер Доступно не для всех категорий товаров"
)
# This one made the generous parser report boots at 15 ₽ and 77 ₽.
BOOTS_ARTICLE_SNIPPET = (
    "Цвет серый, черный.\xa0... Ботинки со скошенным каблуком. "
    "Женская обувь фирмы алми. Казаки 40 размер…\u2060В ПВЗ Курьер "
    "Доступно не для всех категорий товаров"
)

# Google, "site:wildberries.ru кремовый свитер женский"
SWEATER_SIZE_CHART = (
    "XS 40-42; M 44-46; L 46-48. Все размеры. Таблица размеров. "
    "Polo Ralph Lauren. 4,6 22 оценки. 2 201 ₽ 5 500 ₽ −60%"
)
SWEATER_MODEL_SIZE = (
    "Артикул, 387582459. Состав, полиамид 20%; акрил 80%. Цвет, кремовый; "
    "светло-бежевый. Пол, Женский. Размер на модели, 42-48 (one size oversize)"
)
SWEATER_TAG_PRICE = (
    "Свитер женский паутинка из хлопка летняя Этни. -57%. 5 005 ₽ 11 800 ₽ "
    "−57 ... Свитер кремовый оверсайз вязаный"
)
SWEATER_NO_PRICE = (
    "Свитер женский вязаный Lela Ажурный трикотаж 6071533 Кремовый LELA "
    "680123466 в интернет-магазине WildBerries.ru. Бесплатная доставка"
)

# Yandex, "site:wildberries.ru кожаная сумка структурированная"
BAG_TAG_SNIPPET = (
    "Джемпер кремовый женский - Большой выбор в интернет-магазине "
    "WildBerries.ru.Рейтинг 4,2 из 54,2457K отзывов на магазин «...»"
)

# The search index glues a structured price block onto the page text. Read
# naively, "Цена30153 015руб." reports boots at 15 ₽ and 77 ₽.
BOOTS_GLUE_PRICE_SNIPPET = (
    "Цвет серый, черный.\xa0... Ботинки со скошенным каблуком. Женская обувь "
    "фирмы алми. Казаки 40 размер…\u2060В ПВЗ Курьер Доступно не для всех "
    "категорий товаровЦена30153\xa0015руб.₽Скидка -2%−2%вместо30773\xa0077руб."
    "₽При оплате WB\xa0Кошельком\u2060Цена 3\xa0015\xa0₽ действительна при "
    "оплате WB\xa0Кошельком, другими способами"
)
BOOTS_GLUE_PRICE_NO_PROSE = (
    "Достоинства: каблуки красивые. Цвет черный. Вид 5.\u2060В ПВЗ Курьер "
    "Доступно не для всех категорий товаровЦена24942\xa0494руб.₽Скидка -71%−71%"
    "вместо86188\xa0618руб.₽"
)
