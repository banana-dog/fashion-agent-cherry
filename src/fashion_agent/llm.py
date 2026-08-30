import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from dataclasses import dataclass


load_dotenv()

llm = ChatDeepSeek(
    model=os.getenv("MODEL", "deepseek-v4-flash"),
    api_key=os.getenv("API_KEY"), # type: ignore
    temperature=0,
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    },
)

@dataclass
class Context:
    user_id: str
    locale: str = "ru-RU"
    currency: str = "RUB"