from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_id: int
    k: int = Field(default=5, ge=1, le=20)
    exclude_seen: bool = True


class SimilarRequest(BaseModel):
    product_id: int
    k: int = Field(default=5, ge=1, le=20)


class ProductOut(BaseModel):
    product_id: int
    score: float
    title: str | None = None
    brand: str | None = None
    category_code: str | None = None
    price: float | None = None
    reason: str


class RecommendResponse(BaseModel):
    user_id: int
    reason: str
    items: list[ProductOut]


class SimilarResponse(BaseModel):
    product_id: int
    items: list[ProductOut]
