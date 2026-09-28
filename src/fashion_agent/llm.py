import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# An explicit path: relying on find_dotenv makes the key load depend on which
# file happened to trigger the import.
load_dotenv(Path(__file__).parent / ".env")


def drop_unsupported_proxy_env():
    for key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        value = os.environ.get(key)

        if not value:
            continue

        if urlparse(value).scheme == "socks":
            os.environ.pop(
                key,
                None,
            )


drop_unsupported_proxy_env()

llm = ChatDeepSeek(
    model=os.getenv("MODEL", "deepseek-v4-flash"),
    api_key=os.getenv("API_KEY"),  # type: ignore
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}},
)


@dataclass
class Context:
    user_id: str
    locale: str = "ru-RU"
    currency: str = "RUB"
