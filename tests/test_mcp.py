from ecomrec.serving.mcp_server import TOOL_NAMES, build_mcp, registered_tool_names


def test_mcp_sdk_registers_tools():
    mcp = build_mcp()
    names = registered_tool_names(mcp)
    assert names == set(TOOL_NAMES)
