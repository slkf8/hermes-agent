import json

from agent.subagents import (
    STATUS_BLOCKED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    route_named_subagent,
    validate_raw_named_subagent_request,
)


def test_unknown_agent_fails_closed():
    r = route_named_subagent(agent_name="@debug", goal="x", parent_agent=object())
    assert r["status"] == STATUS_BLOCKED


def test_empty_goal_fails_closed():
    r = route_named_subagent(agent_name="@research", goal="", parent_agent=object())
    assert r["status"] == STATUS_BLOCKED


def test_unsafe_toolsets_field_rejected():
    err = validate_raw_named_subagent_request(
        {"agent_name": "@research", "goal": "x", "toolsets": ["terminal"]}
    )
    assert err and "unsafe fields" in err


def test_research_accepts_public_url():
    err = validate_raw_named_subagent_request(
        {
            "agent_name": "@research",
            "goal": "x",
            "input_refs": [
                {"type": "public_url", "ref": "https://example.com", "access": "read_only"}
            ],
        }
    )
    assert err is None


def test_audit_static_rejects_public_url():
    err = validate_raw_named_subagent_request(
        {
            "agent_name": "@audit_static",
            "goal": "x",
            "input_refs": [
                {"type": "public_url", "ref": "https://example.com", "access": "read_only"}
            ],
        }
    )
    assert err


def test_forbidden_refs_rejected():
    for ref in ["/opt/data/auth.json", "/home/user/.env", "/home/user/config.yaml"]:
        err = validate_raw_named_subagent_request(
            {
                "agent_name": "@audit_static",
                "goal": "x",
                "input_refs": [
                    {"type": "allowlisted_file", "ref": ref, "access": "read_only"}
                ],
            },
            allowlisted_roots=["/home/user"],
        )
        assert err


def test_context_secret_like_string_blocked():
    err = validate_raw_named_subagent_request(
        {"agent_name": "@research", "goal": "x", "context": "api_key = abc"}
    )
    assert err


def test_delegate_called_with_leaf_role_and_fixed_toolset():
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return json.dumps({"results": [{"status": "completed", "summary": "ok"}]})

    r = route_named_subagent(
        agent_name="@research",
        goal="x",
        parent_agent=object(),
        _delegate_runner=fake_runner,
    )

    assert r["status"] == STATUS_COMPLETED
    assert calls[0]["role"] == "leaf"
    assert calls[0]["toolsets"] == ["research_leaf"]


def test_non_json_delegate_result_failed():
    def fake_runner(**kwargs):
        return "not json"

    r = route_named_subagent(
        agent_name="@research",
        goal="x",
        parent_agent=object(),
        _delegate_runner=fake_runner,
    )
    assert r["status"] == STATUS_FAILED
