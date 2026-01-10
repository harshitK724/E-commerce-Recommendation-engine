from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query

from ecomrec.models.infer import RecommendationService
from ecomrec.serving.schemas import (
    ProductOut,
    RecommendRequest,
    RecommendResponse,
    SimilarRequest,
    SimilarResponse,
)


def create_app(service: RecommendationService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not getattr(app.state, "service", None):
            app.state.service = service or RecommendationService.load()
        yield

    app = FastAPI(
        title="E-Commerce Recommendation Server",
        version="0.1.0",
        lifespan=lifespan,
    )
    if service is not None:
        app.state.service = service

    def get_service() -> RecommendationService:
        svc = getattr(app.state, "service", None)
        if svc is None:
            raise HTTPException(status_code=503, detail="Recommendation service is not loaded.")
        return svc

    Svc = Annotated[RecommendationService, Depends(get_service)]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/v1/recommendations", response_model=RecommendResponse)
    def recommendations(body: RecommendRequest, svc: Svc) -> RecommendResponse:
        items = svc.recommend(body.user_id, k=body.k, exclude_seen=body.exclude_seen)
        reason = items[0].reason if items else "popularity_fallback"
        return RecommendResponse(
            user_id=body.user_id,
            reason=reason,
            items=[ProductOut.model_validate(it.as_dict()) for it in items],
        )

    @app.get("/v1/users/{user_id}/history")
    def user_history(user_id: int, svc: Svc, n: int = Query(default=10, ge=1, le=50)) -> dict:
        return {"user_id": user_id, "events": svc.history(user_id, n=n)}

    @app.get("/v1/products/{product_id}")
    def product(product_id: int, svc: Svc) -> dict:
        found = svc.get_product(product_id)
        if not found:
            raise HTTPException(status_code=404, detail="Unknown product_id")
        return found

    @app.post("/v1/similar-items", response_model=SimilarResponse)
    def similar_items(body: SimilarRequest, svc: Svc) -> SimilarResponse:
        items = svc.similar_items(body.product_id, k=body.k)
        if not items:
            raise HTTPException(status_code=404, detail="Unknown product_id")
        return SimilarResponse(
            product_id=body.product_id,
            items=[ProductOut.model_validate(it.as_dict()) for it in items],
        )

    return app


app = create_app()
