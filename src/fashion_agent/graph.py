from fashion_agent.nodes import (
    ask_questions, 
    extract_request, 
    ready,
    route_after_extraction,
    update_style_memory,
    load_style_memory
)
from fashion_agent.states import FashionState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import (
    StateGraph,
    START,
    END,
)
from langgraph.store.memory import InMemoryStore


store = InMemoryStore()
checkpointer = InMemorySaver()
builder = StateGraph(FashionState)

builder.add_node(
    "extract_request",
    extract_request
)

builder.add_node(
    "ask_questions",
    ask_questions
)

builder.add_node(
    "ready",
    ready
)

builder.add_node(
    "load_style_memory",
    load_style_memory
)

builder.add_node(
    "update_style_memory",
    update_style_memory
)

builder.add_edge(
    START,
    "update_style_memory"
)

builder.add_edge(
    "update_style_memory",
    "load_style_memory"
)

builder.add_edge(
    "update_style_memory",
    "extract_request"
)

builder.add_conditional_edges(
    "extract_request",
    route_after_extraction,
)

builder.add_edge(
    "ask_questions",
    END
)

builder.add_edge(
    "ready",
    END
)

graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)