"""In-process tools shared with the MCP server."""

from __future__ import annotations

from ecomrec.models.infer import RecommendationService


def get_recommendations(service: RecommendationService, user_id: int, k: int = 5) -> dict:
    items = service.recommend(int(user_id), k=int(k))
    return {"user_id": int(user_id), "items": [it.as_dict() for it in items]}


def get_similar_products(service: RecommendationService, product_id: int, k: int = 5) -> dict:
    items = service.similar_items(int(product_id), k=int(k))
    return {"product_id": int(product_id), "items": [it.as_dict() for it in items]}


def get_user_history(service: RecommendationService, user_id: int, n: int = 10) -> dict:
    return {"user_id": int(user_id), "events": service.history(int(user_id), n=int(n))}


def get_product(service: RecommendationService, product_id: int) -> dict:
    return {"product": service.get_product(int(product_id))}
