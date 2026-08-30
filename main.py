import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.fashion_agent.graph import graph
from src.fashion_agent.nodes import Context

def main():
    thread_id = str(uuid.uuid4())
    config = {
    "configurable": {
        "thread_id": thread_id
    }
    }

    print("🍒 Cherry Pick")
    print("Расскажи, какой образ тебе нужен.\n")

    while True:
        user_input = input("you > ").strip()
        if user_input == "/new":
            thread_id = str(uuid.uuid4())

            config = {
                "configurable": {
                    "thread_id": thread_id
                }
            }

            print("\n✨ Новый разговор\n")
            continue

        if user_input.lower() in {
            "exit",
            "quit",
            "q",
        }:
            break

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=user_input
                    )
                ]
            }, # type: ignore
            config=config, # type: ignore
            context=Context(user_id=thread_id)
        )

        last_message = result["messages"][-1]

        print()
        print("cherry >", last_message.content)
        print()


if __name__ == "__main__":
    main()