"""LangGraph router: route -> (tools | skip) -> grounded reply."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from ecomrec.agent.prompts import (
    GROUNDING_SYSTEM,
    ROUTER_SYSTEM,
    extract_int,
    heuristic_route,
)
from ecomrec.agent.tools import get_product, get_recommendations, get_similar_products, get_user_history
from ecomrec.config import settings
from ecomrec.models.infer import RecommendationService

VALID_ROUTES = {
    "recommend_for_user",
    "similar_to_item",
    "lookup_product",
    "lookup_history",
    "chitchat",
}


class AgentState(TypedDict, total=False):
    message: str
    user_id: int | None
    product_id: int | None
    route: str
    tool_result: dict[str, Any] | None
    reply: str


def _chat(system: str, user: str) -> str | None:
    if not settings.openai_api_key and not settings.openai_base_url:
        return None
    base = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    url = base + "/chat/completions"
    payload = {
        "model": settings.openai_model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if settings.openai_api_key:
        headers["Authorization"] = f"Bearer {settings.openai_api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError):
        return None


def _route_node(state: AgentState) -> AgentState:
    message = state.get("message") or ""
    routed = _chat(ROUTER_SYSTEM, message)
    route = (routed or "").strip().split()[0] if routed else ""
    if route not in VALID_ROUTES:
        route = heuristic_route(message)
    user_id = state.get("user_id") or extract_int(message, ("user_id", "user"))
    product_id = state.get("product_id") or extract_int(message, ("product_id", "product", "item"))
    return {**state, "route": route, "user_id": user_id, "product_id": product_id}


def _after_router(state: AgentState) -> Literal["tools", "ground"]:
    if state.get("route") == "chitchat":
        return "ground"
    return "tools"


def _tool_node(state: AgentState, service: RecommendationService) -> AgentState:
    route = state.get("route") or "chitchat"
    result: dict[str, Any] | None = None
    if route == "recommend_for_user":
        uid = state.get("user_id")
        result = (
            {"error": "user_id is required to recommend products"}
            if uid is None
            else get_recommendations(service, uid, k=5)
        )
    elif route == "similar_to_item":
        pid = state.get("product_id")
        result = (
            {"error": "product_id is required for similar items"}
            if pid is None
            else get_similar_products(service, pid, k=5)
        )
    elif route == "lookup_product":
        pid = state.get("product_id")
        result = {"error": "product_id is required"} if pid is None else get_product(service, pid)
    elif route == "lookup_history":
        uid = state.get("user_id")
        result = {"error": "user_id is required"} if uid is None else get_user_history(service, uid, n=10)
    return {**state, "tool_result": result}


def _format_items(items: list[dict]) -> str:
    lines = []
    for i, it in enumerate(items, start=1):
        title = it.get("title") or f"product {it.get('product_id')}"
        price = it.get("price")
        price_s = f" ${price:.2f}" if isinstance(price, (int, float)) else ""
        lines.append(f"{i}. {title}{price_s} (id={it.get('product_id')}, reason={it.get('reason')})")
    return "\n".join(lines)


def _ground_node(state: AgentState) -> AgentState:
    route = state.get("route") or "chitchat"
    message = state.get("message") or ""
    result = state.get("tool_result")
    if route == "chitchat":
        return {
            **state,
            "reply": (
                "I can recommend products for a user_id, find similar items, or look up a product. "
                "Try: 'recommend something for user_id 12'."
            ),
        }
    if result is not None:
        llm_reply = _chat(GROUNDING_SYSTEM, f"USER: {message}\nTOOL_RESULTS: {result}")
        if llm_reply:
            return {**state, "reply": llm_reply}
    if not result:
        return {**state, "reply": "I could not retrieve recommendations."}
    if result.get("error"):
        return {**state, "reply": str(result["error"])}
    if result.get("items"):
        header = "Here are personalized picks grounded in the recommendation model:\n"
        return {**state, "reply": header + _format_items(result["items"])}
    if result.get("product"):
        return {**state, "reply": f"Product details: {result['product']}"}
    if result.get("events"):
        return {**state, "reply": f"Recent history: {result['events'][:5]}"}
    return {**state, "reply": "No matching products were returned by the model."}


def build_graph(service: RecommendationService):
    graph = StateGraph(AgentState)
    graph.add_node("router", _route_node)
    graph.add_node("tools", lambda s: _tool_node(s, service))
    graph.add_node("ground", _ground_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _after_router,
        {"tools": "tools", "ground": "ground"},
    )
    graph.add_edge("tools", "ground")
    graph.add_edge("ground", END)
    return graph.compile()


def run_turn(
    service: RecommendationService,
    message: str,
    user_id: int | None = None,
    product_id: int | None = None,
) -> AgentState:
    return build_graph(service).invoke(
        {"message": message, "user_id": user_id, "product_id": product_id}
    )
