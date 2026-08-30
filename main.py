from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.fashion_agent.graph import graph

def main():
    config = {
        "configurable": {
            "thread_id": "demo-user"
        }
    }

    print("🍒 Cherry Pick")
    print("Расскажи, какой образ тебе нужен.\n")

    while True:
        user_input = input("you > ").strip()

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
        )

        last_message = result["messages"][-1]

        print()
        print("cherry >", last_message.content)
        print()


if __name__ == "__main__":
    main()