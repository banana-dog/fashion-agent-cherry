from dotenv import load_dotenv
from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from fashion_agent.basic_requests import (
    ask_questions,
    extract_request,
    route_after_extraction,
    talk,
)
from fashion_agent.knowledge.nodes import (
    interpret_style,
    retrieve_style_knowledge,
)
from fashion_agent.outfit_builder import (
    build_outfits,
    critique_outfits,
    present_outfits,
    route_after_build,
)
from fashion_agent.product_search.product_search import (
    create_search_plan,
    dispatch_product_searches,
    search_one_category,
)
from fashion_agent.product_search.products_processing import (
    enrich_product_attributes,
    rank_products,
)
from fashion_agent.states import FashionState
from fashion_agent.storage import build_checkpointer, build_store
from fashion_agent.style_dna import load_style_memory, update_style_memory
from fashion_agent.tool_node import check_context

load_dotenv()

store = build_store()
checkpointer = build_checkpointer()
builder = StateGraph(FashionState)

############NODES##############

builder.add_node("extract_request", extract_request)

builder.add_node("ask_questions", ask_questions)

builder.add_node("talk", talk)

builder.add_node(
    "load_style_memory",
    load_style_memory,  # type: ignore
)

builder.add_node(
    "update_style_memory",
    update_style_memory,  # type: ignore
)

builder.add_node(
    "create_search_plan",
    create_search_plan,  # type: ignore
)

builder.add_node(
    "retrieve_style_knowledge",
    retrieve_style_knowledge,
)

builder.add_node(
    "check_context",
    check_context,  # type: ignore
)

builder.add_node(
    "interpret_style",
    interpret_style,  # type: ignore
)

builder.add_node(
    "search_one_category",
    search_one_category,  # type: ignore
)

builder.add_node(
    "rank_products",
    rank_products,
)

builder.add_node(
    "build_outfits",
    build_outfits,
)

builder.add_node(
    "critique_outfits",
    critique_outfits,
)

builder.add_node(
    "present_outfits",
    present_outfits,
)

builder.add_node(
    "enrich_product_attributes",
    enrich_product_attributes,
)
#########EDGES##############

builder.add_edge(START, "update_style_memory")

builder.add_edge("update_style_memory", "load_style_memory")

builder.add_edge("load_style_memory", "extract_request")

builder.add_conditional_edges(
    "extract_request",
    route_after_extraction,
)

builder.add_edge("ask_questions", END)

builder.add_edge("talk", END)

# Outside knowledge is gathered before the outfit is planned, so a forecast can
# change what gets searched for.
builder.add_edge("check_context", "retrieve_style_knowledge")

builder.add_edge(
    "retrieve_style_knowledge",
    "interpret_style",
)

builder.add_edge(
    "interpret_style",
    "create_search_plan",
)

builder.add_conditional_edges(
    "create_search_plan",
    dispatch_product_searches,
    [
        "search_one_category",
    ],
)

builder.add_edge(
    "search_one_category",
    "enrich_product_attributes",
)

builder.add_edge(
    "enrich_product_attributes",
    "rank_products",
)

builder.add_edge(
    "rank_products",
    "build_outfits",
)

builder.add_conditional_edges(
    "build_outfits",
    route_after_build,
)

builder.add_edge(
    "critique_outfits",
    "present_outfits",
)

builder.add_edge(
    "present_outfits",
    END,
)

graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)
