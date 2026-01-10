from fastapi.testclient import TestClient

from ecomrec.serving.api import create_app


def test_health_and_recommendations(toy_service):
    app = create_app(service=toy_service)
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"

    known_user = int(toy_service.user_map.iloc[0]["user_id"])
    resp = client.post("/v1/recommendations", json={"user_id": known_user, "k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reason"] == "collaborative"
    assert 1 <= len(body["items"]) <= 5
    assert "product_id" in body["items"][0]

    unknown = client.post("/v1/recommendations", json={"user_id": 999_999_999, "k": 3})
    assert unknown.status_code == 200
    assert unknown.json()["reason"] == "popularity_fallback"
    assert len(unknown.json()["items"]) >= 1


def test_product_and_similar(toy_service):
    app = create_app(service=toy_service)
    client = TestClient(app)
    pid = int(toy_service.item_map.iloc[0]["product_id"])
    product = client.get(f"/v1/products/{pid}")
    assert product.status_code == 200
    similar = client.post("/v1/similar-items", json={"product_id": pid, "k": 3})
    assert similar.status_code == 200
    assert similar.json()["items"]
    missing = client.get("/v1/products/1")
    assert missing.status_code == 404


def test_history_endpoint(toy_service):
    app = create_app(service=toy_service)
    client = TestClient(app)
    uid = int(toy_service.user_map.iloc[0]["user_id"])
    resp = client.get(f"/v1/users/{uid}/history")
    assert resp.status_code == 200
    assert "events" in resp.json()
