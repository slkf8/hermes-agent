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


MVP2_BATCH1_AGENTS = ["@debug_static", "@test_planner", "@patch_reviewer"]


def test_mvp2_batch1_routes_with_leaf_role_and_static_analysis_toolset():
    for agent_name in MVP2_BATCH1_AGENTS:
        calls = []

        def fake_runner(**kwargs):
            calls.append(kwargs)
            return json.dumps({"results": [{"status": "completed", "summary": "ok"}]})

        r = route_named_subagent(
            agent_name=agent_name,
            goal="x",
            parent_agent=object(),
            _delegate_runner=fake_runner,
        )

        assert r["status"] == STATUS_COMPLETED, agent_name
        assert calls[0]["role"] == "leaf", agent_name
        assert calls[0]["toolsets"] == ["static_analysis_leaf"], agent_name


def test_mvp2_batch1_rejects_public_url():
    for agent_name in MVP2_BATCH1_AGENTS:
        err = validate_raw_named_subagent_request(
            {
                "agent_name": agent_name,
                "goal": "x",
                "input_refs": [
                    {"type": "public_url", "ref": "https://example.com", "access": "read_only"}
                ],
            }
        )
        assert err, agent_name


def test_mvp2_batch1_rejects_unsafe_toolsets_field():
    for agent_name in MVP2_BATCH1_AGENTS:
        err = validate_raw_named_subagent_request(
            {"agent_name": agent_name, "goal": "x", "toolsets": ["terminal"]}
        )
        assert err and "unsafe fields" in err, agent_name


def test_mvp2_batch1_rejects_forbidden_refs():
    for agent_name in MVP2_BATCH1_AGENTS:
        for ref in ["/opt/data/auth.json", "/home/user/.env", "/home/user/config.yaml"]:
            err = validate_raw_named_subagent_request(
                {
                    "agent_name": agent_name,
                    "goal": "x",
                    "input_refs": [
                        {"type": "allowlisted_file", "ref": ref, "access": "read_only"}
                    ],
                },
                allowlisted_roots=["/home/user"],
            )
            assert err, f"{agent_name} {ref}"


def test_mvp2_batch1_empty_goal_fails_closed():
    for agent_name in MVP2_BATCH1_AGENTS:
        r = route_named_subagent(agent_name=agent_name, goal="", parent_agent=object())
        assert r["status"] == STATUS_BLOCKED, agent_name


def test_mvp2_batch1_missing_parent_agent_fails_closed():
    for agent_name in MVP2_BATCH1_AGENTS:
        r = route_named_subagent(agent_name=agent_name, goal="x", parent_agent=None)
        assert r["status"] == STATUS_BLOCKED, agent_name


def test_mvp2_batch1_allows_allowlisted_file_ref():
    for agent_name in MVP2_BATCH1_AGENTS:
        err = validate_raw_named_subagent_request(
            {
                "agent_name": agent_name,
                "goal": "x",
                "input_refs": [
                    {"type": "provided_text", "ref": "some snippet", "access": "read_only"}
                ],
            }
        )
        assert err is None, agent_name


SECURITY_GATE_AGENT = "@security_gate"


def test_security_gate_exists_in_presets():
    from agent.subagents import _PRESETS

    assert SECURITY_GATE_AGENT in _PRESETS
    assert _PRESETS[SECURITY_GATE_AGENT].toolsets == ("static_analysis_leaf",)


def test_security_gate_routes_with_leaf_role_and_static_analysis_toolset():
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return json.dumps({"results": [{"status": "completed", "summary": "ok"}]})

    r = route_named_subagent(
        agent_name=SECURITY_GATE_AGENT,
        goal="review evidence",
        parent_agent=object(),
        _delegate_runner=fake_runner,
    )

    assert r["status"] == STATUS_COMPLETED
    assert calls[0]["role"] == "leaf"
    assert calls[0]["toolsets"] == ["static_analysis_leaf"]


def test_security_gate_unknown_variant_blocked():
    r = route_named_subagent(agent_name="@security", goal="x", parent_agent=object())
    assert r["status"] == STATUS_BLOCKED


def test_security_gate_empty_goal_fails_closed():
    r = route_named_subagent(agent_name=SECURITY_GATE_AGENT, goal="", parent_agent=object())
    assert r["status"] == STATUS_BLOCKED


def test_security_gate_whitespace_goal_fails_closed():
    r = route_named_subagent(agent_name=SECURITY_GATE_AGENT, goal="   ", parent_agent=object())
    assert r["status"] == STATUS_BLOCKED


def test_security_gate_missing_parent_agent_fails_closed():
    r = route_named_subagent(agent_name=SECURITY_GATE_AGENT, goal="x", parent_agent=None)
    assert r["status"] == STATUS_BLOCKED


def test_security_gate_rejects_unsafe_toolsets_field():
    err = validate_raw_named_subagent_request(
        {"agent_name": SECURITY_GATE_AGENT, "goal": "x", "toolsets": ["terminal"]}
    )
    assert err and "unsafe fields" in err


def test_security_gate_rejects_public_url():
    err = validate_raw_named_subagent_request(
        {
            "agent_name": SECURITY_GATE_AGENT,
            "goal": "x",
            "input_refs": [
                {"type": "public_url", "ref": "https://example.com", "access": "read_only"}
            ],
        }
    )
    assert err


def test_security_gate_rejects_forbidden_refs():
    for ref in ["/opt/data/auth.json", "/home/user/.env", "/home/user/config.yaml"]:
        err = validate_raw_named_subagent_request(
            {
                "agent_name": SECURITY_GATE_AGENT,
                "goal": "x",
                "input_refs": [
                    {"type": "allowlisted_file", "ref": ref, "access": "read_only"}
                ],
            },
            allowlisted_roots=["/home/user"],
        )
        assert err, ref


def test_security_gate_allows_provided_text():
    err = validate_raw_named_subagent_request(
        {
            "agent_name": SECURITY_GATE_AGENT,
            "goal": "x",
            "input_refs": [
                {"type": "provided_text", "ref": "some snippet", "access": "read_only"}
            ],
        }
    )
    assert err is None


def test_security_gate_non_json_runner_failed():
    def fake_runner(**kwargs):
        return "not json"

    r = route_named_subagent(
        agent_name=SECURITY_GATE_AGENT,
        goal="x",
        parent_agent=object(),
        _delegate_runner=fake_runner,
    )
    assert r["status"] == STATUS_FAILED
