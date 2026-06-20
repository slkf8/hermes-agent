from toolsets import TOOLSETS


def _tools(name):
    return set(TOOLSETS[name]["tools"])


def test_research_leaf_static_scope_only_public_research_tools():
    assert _tools("research_leaf") == {"web_search", "web_extract"}


def test_audit_static_leaf_static_scope_only_read_search_file_tools():
    assert _tools("audit_static_leaf") == {"read_file", "search_files"}


def test_static_analysis_leaf_static_scope_only_read_search_file_tools():
    assert _tools("static_analysis_leaf") == {"read_file", "search_files"}


def test_named_subagent_static_scopes_exclude_delegate_task():
    assert "delegate_task" not in _tools("research_leaf")
    assert "delegate_task" not in _tools("audit_static_leaf")
    assert "delegate_task" not in _tools("static_analysis_leaf")


def test_named_subagent_static_scopes_exclude_write_and_execute_tools():
    denied = {"terminal", "process", "execute_code", "write_file", "patch"}
    assert _tools("research_leaf").isdisjoint(denied)
    assert _tools("audit_static_leaf").isdisjoint(denied)
    assert _tools("static_analysis_leaf").isdisjoint(denied)
