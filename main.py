import argparse
import uuid

from langchain_core.messages import HumanMessage

from src.fashion_agent.graph import graph
from src.fashion_agent.styleDNA import Context
from src.fashion_agent.web import run_web_server


def run_cli():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    context = Context(
        user_id="demo-user",
        locale="ru-RU",
        currency="RUB",
    )
    print("🍒 Cherry Pick")
    print("Расскажи, какой образ тебе нужен.\n")

    while True:
        user_input = input("you > ").strip()
        if user_input == "/new":
            thread_id = str(uuid.uuid4())

            config = {"configurable": {"thread_id": thread_id}}

            print("\n✨ Новый разговор\n")
            continue

        if user_input.lower() in {
            "exit",
            "quit",
            "q",
        }:
            break

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},  # type: ignore
            config=config,  # type: ignore
            context=context,  # type: ignore
        )

        last_message = result["messages"][-1]

        print()
        print("cherry >", last_message.content)
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        choices=[
            "cli",
            "web",
        ],
        default="cli",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )
    args = parser.parse_args()

    if args.mode == "web":
        run_web_server(
            host=args.host,
            port=args.port,
        )
        return

    run_cli()


if __name__ == "__main__":
    main()
