from ecomrec.agent.graph import run_turn
from ecomrec.agent.prompts import heuristic_route


def test_heuristic_routes_recommend():
    assert heuristic_route("recommend me something") == "recommend_for_user"
    assert heuristic_route("more like this product") == "similar_to_item"


def test_graph_recommend_route(toy_service):
    user_id = int(toy_service.user_map.iloc[0]["user_id"])
    state = run_turn(toy_service, "recommend me something", user_id=user_id)
    assert state["route"] == "recommend_for_user"
    assert state["tool_result"] is not None
    assert state["tool_result"]["items"]
    assert "personalized" in state["reply"].lower() or "id=" in state["reply"]


def test_graph_requires_user_id(toy_service):
    state = run_turn(toy_service, "recommend me something")
    assert state["route"] == "recommend_for_user"
    assert state["tool_result"]["error"]


def test_graph_chitchat_skips_tools(toy_service):
    state = run_turn(toy_service, "hello there")
    assert state["route"] == "chitchat"
    assert state.get("tool_result") is None
    assert "recommend" in state["reply"].lower()
