"""MCP tool server (official SDK) wrapping RecommendationService."""

from __future__ import annotations

from ecomrec.models.infer import RecommendationService

_SERVICE: RecommendationService | None = None

TOOL_NAMES = (
    "get_recommendations",
    "get_similar_products",
    "get_user_history",
    "get_product",
)


def get_mcp_service() -> RecommendationService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = RecommendationService.load()
    return _SERVICE


def tool_get_recommendations(user_id: int, k: int = 5) -> dict:
    items = get_mcp_service().recommend(int(user_id), k=int(k))
    return {"user_id": int(user_id), "items": [it.as_dict() for it in items]}


def tool_get_similar_products(product_id: int, k: int = 5) -> dict:
    items = get_mcp_service().similar_items(int(product_id), k=int(k))
    return {"product_id": int(product_id), "items": [it.as_dict() for it in items]}


def tool_get_user_history(user_id: int, n: int = 10) -> dict:
    return {"user_id": int(user_id), "events": get_mcp_service().history(int(user_id), n=int(n))}


def tool_get_product(product_id: int) -> dict:
    return {"product": get_mcp_service().get_product(int(product_id))}


def build_mcp():
    from mcp.server import MCPServer

    mcp = MCPServer(
        "ecomrec",
        instructions=(
            "E-commerce recommendation tools. Call get_recommendations for a shopper, "
            "get_similar_products for item-item, get_user_history for recent events, "
            "and get_product for catalog metadata. Do not invent product ids."
        ),
    )

    @mcp.tool()
    def get_recommendations(user_id: int, k: int = 5) -> dict:
        """Return personalized product recommendations for a shopper user_id."""
        return tool_get_recommendations(user_id, k)

    @mcp.tool()
    def get_similar_products(product_id: int, k: int = 5) -> dict:
        """Return items similar to product_id using matrix-factorization embeddings."""
        return tool_get_similar_products(product_id, k)

    @mcp.tool()
    def get_user_history(user_id: int, n: int = 10) -> dict:
        """Return recent interaction history for a shopper."""
        return tool_get_user_history(user_id, n)

    @mcp.tool()
    def get_product(product_id: int) -> dict:
        """Look up catalog metadata for a product_id."""
        return tool_get_product(product_id)

    return mcp


def registered_tool_names(mcp) -> set[str]:
    return {info.name for info in mcp._tool_manager.list_tools()}


def run_mcp() -> None:
    build_mcp().run()
