from toolsets import TOOLSETS


def _tools(name):
    return set(TOOLSETS[name]["tools"])


def test_research_leaf_exact_tools():
    assert _tools("research_leaf") == {"web_search", "web_extract"}


def test_audit_static_leaf_exact_tools():
    assert _tools("audit_static_leaf") == {"read_file", "search_files"}


def test_research_leaf_excludes_dangerous_tools():
    denied = {
        "delegate_task",
        "terminal",
        "process",
        "execute_code",
        "write_file",
        "patch",
        "memory",
        "skill_manage",
        "send_message",
        "cronjob",
    }
    assert _tools("research_leaf").isdisjoint(denied)


def test_audit_static_leaf_excludes_dangerous_and_web_tools():
    denied = {
        "delegate_task",
        "terminal",
        "process",
        "execute_code",
        "write_file",
        "patch",
        "memory",
        "skill_manage",
        "send_message",
        "cronjob",
        "web_search",
        "web_extract",
    }
    assert _tools("audit_static_leaf").isdisjoint(denied)


def test_static_analysis_leaf_exact_tools():
    assert _tools("static_analysis_leaf") == {"read_file", "search_files"}


def test_static_analysis_leaf_excludes_dangerous_and_web_tools():
    denied = {
        "delegate_task",
        "terminal",
        "process",
        "execute_code",
        "write_file",
        "patch",
        "memory",
        "skill_manage",
        "send_message",
        "cronjob",
        "web_search",
        "web_extract",
    }
    assert _tools("static_analysis_leaf").isdisjoint(denied)
