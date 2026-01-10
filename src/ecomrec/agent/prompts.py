ROUTER_SYSTEM = """You route shopping messages to exactly one tool name.
Valid routes: recommend_for_user, similar_to_item, lookup_product, chitchat.
Reply with only the route name.
"""

GROUNDING_SYSTEM = """You are a shopping assistant. You may ONLY mention products in TOOL_RESULTS.
Never invent product ids, brands, or prices. If TOOL_RESULTS is empty, say you have no matches.
Be concise and helpful.
"""


def heuristic_route(message: str) -> str:
    text = message.lower()
    if any(w in text for w in ("similar", "like this", "more like", "related")):
        return "similar_to_item"
    if any(w in text for w in ("history", "what did i", "past purchase", "viewed")):
        return "lookup_history"
    if any(w in text for w in ("recommend", "suggest", "for me", "show me", "gift")):
        return "recommend_for_user"
    if "product" in text or "price" in text or "brand" in text:
        return "lookup_product"
    return "chitchat"


def extract_int(message: str, keys: tuple[str, ...]) -> int | None:
    import re

    for key in keys:
        match = re.search(rf"{key}\s*[=:#]?\s*(\d+)", message, flags=re.I)
        if match:
            return int(match.group(1))
    match = re.search(r"\b(\d{3,})\b", message)
    return int(match.group(1)) if match else None
