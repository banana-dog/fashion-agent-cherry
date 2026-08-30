from fashion_agent.basic_requests import (
    ask_questions, 
    extract_request, 
    ready,
    route_after_extraction
)
from src.fashion_agent.styleDNA import (
    update_style_memory,
    load_style_memory
)
from src.fashion_agent.search_planner import(
    create_search_plan,
    execute_searches,
    rank_products,
    present_candidates
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

############NODES##############

builder.add_node(
    "extract_request",
    extract_request
)

builder.add_node(
    "ask_questions",
    ask_questions
)

builder.add_node(
    "load_style_memory",
    load_style_memory  # type: ignore
)

builder.add_node(
    "update_style_memory",
    update_style_memory # type: ignore
)

builder.add_node(
    "create_search_plan",
    create_search_plan,
)

builder.add_node(
    "execute_searches",
    execute_searches,
)

builder.add_node(
    "present_candidates",
    present_candidates,
)

builder.add_node(
    "rank_products",
    rank_products,
)
#########EDGES##############

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
    "create_search_plan",
    "execute_searches",
)

builder.add_edge(
    "execute_searches",
    "rank_products",
)

builder.add_edge(
    "rank_products",
    "present_candidates",
)

builder.add_edge(
    "present_candidates",
    END,
)

graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)