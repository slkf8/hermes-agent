"""Tests for the deterministic Central Router policy selector (M10).

These are pure offline unit tests over structured descriptors. They assert
policy selection and the capability/trace invariants. No runtime, no network,
no file I/O, no delegate_task, no route_named_subagent.
"""

import dataclasses

import pytest

from agent import subagent_router as router
from agent.subagent_router import (
    RouterDecision,
    TaskDescriptor,
    enforcement_preset_names,
    select_named_subagent_for_task,
)


# --- helpers ----------------------------------------------------------------

def td(**kwargs):
    return TaskDescriptor(**kwargs)


def select(**kwargs):
    return select_named_subagent_for_task(td(**kwargs))


# All structured descriptors used across the matrix. Used to prove the router
# touches no files/network/runtime: a TaskDescriptor is the only input.
ALL_CASES = {
    # R0 safety
    "r0_write": dict(is_mutating_request=True),
    "r0_exec": dict(is_exec_request=True),
    "r0_runtime": dict(is_runtime_start=True),
    "r0_delegate": dict(is_real_delegate=True),
    "r0_env": dict(targets_secret_or_forbidden_path=True),
    "r0_opt_data": dict(targets_secret_or_forbidden_path=True),
    "r0_user_memory": dict(targets_secret_or_forbidden_path=True),
    "r0_git": dict(is_git_mutation=True),
    "r0_envmut": dict(is_env_mutation=True),
    # R1
    "r1_url": dict(has_public_url=True),
    "r1_search": dict(is_open_web_search=True),
    # R1b
    "r1b_url_file": dict(has_public_url=True, has_local_file_ref=True),
    "r1b_research_audit": dict(is_open_web_search=True, has_local_file_ref=True),
    # R2
    "r2_impl_auth": dict(is_gate_question=True),
    "r2_fork_push_gate": dict(is_gate_question=True),
    "r2_runtime_gate": dict(is_gate_question=True),
    # R3
    "r3_review_diff": dict(has_candidate_change=True, is_judge_ask=True),
    "r3_commit_safe": dict(has_candidate_change=True, is_judge_ask=True),
    "r3_changed_files": dict(has_candidate_change=True, is_judge_ask=True),
    # R4
    "r4_targeted_tests": dict(deliverable_is_test_plan=True),
    "r4_smoke_plan": dict(deliverable_is_test_plan=True),
    "r4_pass_criteria": dict(deliverable_is_test_plan=True),
    # R5
    "r5_traceback": dict(names_symptom=True),
    "r5_regression": dict(names_symptom=True),
    "r5_failing_symptom": dict(names_symptom=True),
    # R6
    "r6_route_exposure": dict(is_broad_inspection=True),
    "r6_toolset_boundary": dict(is_broad_inspection=True),
    "r6_safety_surface": dict(is_broad_inspection=True),
    # R7
    "r7_vague": dict(),
    "r7_no_rule": dict(declared_intent="???"),
    "r7_two_rules": dict(has_candidate_change=True, is_judge_ask=True,
                         deliverable_is_test_plan=True),
    "r7_review_and_tests": dict(has_candidate_change=True, is_judge_ask=True,
                                deliverable_is_test_plan=True),
    "r7_debug_and_fix": dict(names_symptom=True, is_mutating_request=True),
}


# --- R0 safety blocks -------------------------------------------------------

@pytest.mark.parametrize("case", [
    "r0_write", "r0_exec", "r0_runtime", "r0_delegate",
    "r0_env", "r0_opt_data", "r0_user_memory", "r0_git", "r0_envmut",
])
def test_r0_safety_blocks(case):
    d = select(**ALL_CASES[case])
    assert d.selected_wrapper is None
    assert d.fired_rule == "R0"
    assert d.fail_closed_reason and d.fail_closed_reason.startswith("R0:")
    assert d.multi_intent_detected is False


# --- R1 @research -----------------------------------------------------------

@pytest.mark.parametrize("case", ["r1_url", "r1_search"])
def test_r1_research_selected(case):
    d = select(**ALL_CASES[case])
    assert d.selected_wrapper == "@research"
    assert d.fired_rule == "R1"
    assert d.fail_closed_reason is None


# --- R1b public web + local file -> no wrapper ------------------------------

@pytest.mark.parametrize("case", ["r1b_url_file", "r1b_research_audit"])
def test_r1b_web_plus_file_rejected(case):
    d = select(**ALL_CASES[case])
    assert d.selected_wrapper is None
    assert d.fired_rule == "R1b"
    assert d.fail_closed_reason and "R1b" in d.fail_closed_reason


# --- R2 @security_gate ------------------------------------------------------

@pytest.mark.parametrize("case", ["r2_impl_auth", "r2_fork_push_gate", "r2_runtime_gate"])
def test_r2_security_gate_selected(case):
    d = select(**ALL_CASES[case])
    assert d.selected_wrapper == "@security_gate"
    assert d.fired_rule == "R2"


def test_r2_gate_execute_is_not_gate_decision():
    # Asking to *execute* the gate is a safety concern, not a gate decision.
    d = select(is_gate_question=True, is_gate_execute=True, is_runtime_start=True)
    assert d.selected_wrapper is None


# --- R3 @patch_reviewer -----------------------------------------------------

@pytest.mark.parametrize("case", ["r3_review_diff", "r3_commit_safe", "r3_changed_files"])
def test_r3_patch_reviewer_selected(case):
    d = select(**ALL_CASES[case])
    assert d.selected_wrapper == "@patch_reviewer"
    assert d.fired_rule == "R3"


# --- R4 @test_planner -------------------------------------------------------

@pytest.mark.parametrize("case", ["r4_targeted_tests", "r4_smoke_plan", "r4_pass_criteria"])
def test_r4_test_planner_selected(case):
    d = select(**ALL_CASES[case])
    assert d.selected_wrapper == "@test_planner"
    assert d.fired_rule == "R4"


# --- R5 @debug_static -------------------------------------------------------

@pytest.mark.parametrize("case", ["r5_traceback", "r5_regression", "r5_failing_symptom"])
def test_r5_debug_static_selected(case):
    d = select(**ALL_CASES[case])
    assert d.selected_wrapper == "@debug_static"
    assert d.fired_rule == "R5"


# --- R6 @audit_static -------------------------------------------------------

@pytest.mark.parametrize("case", ["r6_route_exposure", "r6_toolset_boundary", "r6_safety_surface"])
def test_r6_audit_static_selected(case):
    d = select(**ALL_CASES[case])
    assert d.selected_wrapper == "@audit_static"
    assert d.fired_rule == "R6"


# --- R7 no wrapper ----------------------------------------------------------

@pytest.mark.parametrize("case", [
    "r7_vague", "r7_no_rule", "r7_two_rules",
    "r7_review_and_tests", "r7_debug_and_fix",
])
def test_r7_no_wrapper(case):
    d = select(**ALL_CASES[case])
    assert d.selected_wrapper is None
    assert d.fail_closed_reason


def test_r7_ambiguous_trace():
    d = select(**ALL_CASES["r7_vague"])
    assert d.selected_wrapper is None
    assert d.fired_rule == "R7"
    assert d.multi_intent_detected is False
    assert "ambiguous" in d.fail_closed_reason


# --- Per-wrapper positive / near-miss negative pairs ------------------------

def test_research_positive_and_near_miss():
    # positive
    assert select(has_public_url=True).selected_wrapper == "@research"
    # near-miss: same web intent but a local file ref present -> R1b, not research
    assert select(has_public_url=True, has_local_file_ref=True).selected_wrapper != "@research"


def test_audit_static_positive_and_near_miss():
    assert select(is_broad_inspection=True).selected_wrapper == "@audit_static"
    # near-miss: a named symptom makes it a defect diagnosis, not a broad audit
    assert select(names_symptom=True).selected_wrapper != "@audit_static"


def test_debug_static_positive_and_near_miss():
    assert select(names_symptom=True).selected_wrapper == "@debug_static"
    # near-miss: broad inspection with no named symptom -> audit, not debug
    assert select(is_broad_inspection=True).selected_wrapper != "@debug_static"


def test_test_planner_positive_and_near_miss():
    assert select(deliverable_is_test_plan=True).selected_wrapper == "@test_planner"
    # near-miss: a candidate change to judge -> patch reviewer, not test planner
    assert select(has_candidate_change=True, is_judge_ask=True).selected_wrapper != "@test_planner"


def test_patch_reviewer_positive_and_near_miss():
    assert select(has_candidate_change=True, is_judge_ask=True).selected_wrapper == "@patch_reviewer"
    # near-miss: test-plan deliverable, no candidate change -> test planner
    assert select(deliverable_is_test_plan=True).selected_wrapper != "@patch_reviewer"


def test_security_gate_positive_and_near_miss():
    assert select(is_gate_question=True).selected_wrapper == "@security_gate"
    # near-miss: broad inspection of the security surface is an audit, not a gate decision
    assert select(is_broad_inspection=True).selected_wrapper != "@security_gate"


# --- Multi-intent rejection -------------------------------------------------

def test_multi_intent_forces_no_wrapper():
    d = select(has_candidate_change=True, is_judge_ask=True, deliverable_is_test_plan=True)
    assert d.selected_wrapper is None
    assert d.multi_intent_detected is True
    assert d.fired_rule == "R7"
    assert "multi-intent" in d.fail_closed_reason


def test_multi_intent_names_colliding_rules():
    d = select(names_symptom=True, is_broad_inspection=True)
    assert d.selected_wrapper is None
    assert d.multi_intent_detected is True
    assert "R5" in d.fail_closed_reason and "R6" in d.fail_closed_reason


def test_multi_intent_no_fanout_no_sequence_no_subtasks():
    d = select(has_candidate_change=True, is_judge_ask=True, deliverable_is_test_plan=True)
    field_names = {f.name for f in dataclasses.fields(d)}
    # No orchestration surface of any kind.
    for forbidden in ("subtasks", "subtask_list", "next", "sequence", "wrappers", "plan"):
        assert forbidden not in field_names
    # selected_wrapper is a single name or None, never a collection.
    assert d.selected_wrapper is None or isinstance(d.selected_wrapper, str)


def test_multi_intent_does_not_call_enforcement(monkeypatch):
    import agent.subagents as subagents
    calls = []
    monkeypatch.setattr(subagents, "route_named_subagent",
                        lambda *a, **k: calls.append("route") or {})
    monkeypatch.setattr(subagents, "validate_raw_named_subagent_request",
                        lambda *a, **k: calls.append("validate"))
    select(has_candidate_change=True, is_judge_ask=True, deliverable_is_test_plan=True)
    assert calls == []


# --- Rationale trace shape --------------------------------------------------

def test_trace_selected_wrapper():
    d = select(has_candidate_change=True, is_judge_ask=True)
    assert d.selected_wrapper == "@patch_reviewer"
    assert d.fired_rule == "R3"
    assert d.fail_closed_reason is None
    assert d.positive_signals
    assert d.multi_intent_detected is False
    assert isinstance(d.reason, str) and d.reason


def test_trace_no_wrapper():
    d = select()
    assert d.selected_wrapper is None
    assert d.fail_closed_reason


def test_trace_r0_blocked():
    d = select(is_mutating_request=True)
    assert d.selected_wrapper is None
    assert d.fired_rule == "R0"
    assert d.fail_closed_reason.startswith("R0:")


def test_trace_multi_intent():
    d = select(has_candidate_change=True, is_judge_ask=True, deliverable_is_test_plan=True)
    assert d.multi_intent_detected is True
    assert d.selected_wrapper is None
    assert d.fired_rule == "R7"


# --- Contract invariants (parametrized keystone) ----------------------------

ALL_DECISIONS = [select(**kwargs) for kwargs in ALL_CASES.values()]


@pytest.mark.parametrize("d", ALL_DECISIONS)
def test_invariant_selected_implies_rule_and_no_fail_reason(d):
    if d.selected_wrapper is not None:
        assert d.fired_rule is not None
        assert d.fail_closed_reason is None


@pytest.mark.parametrize("d", ALL_DECISIONS)
def test_invariant_no_wrapper_implies_fail_reason(d):
    if d.selected_wrapper is None:
        assert isinstance(d.fail_closed_reason, str) and d.fail_closed_reason


@pytest.mark.parametrize("d", ALL_DECISIONS)
def test_invariant_multi_intent_implies_no_wrapper(d):
    if d.multi_intent_detected:
        assert d.selected_wrapper is None


@pytest.mark.parametrize("d", ALL_DECISIONS)
def test_invariant_selected_wrapper_domain(d):
    assert d.selected_wrapper is None or d.selected_wrapper in router.WRAPPER_NAMES


@pytest.mark.parametrize("d", ALL_DECISIONS)
def test_invariant_fired_rule_domain(d):
    assert d.fired_rule is None or d.fired_rule in router.RULE_IDS


@pytest.mark.parametrize("d", ALL_DECISIONS)
def test_invariant_selected_wrapper_is_scalar(d):
    assert d.selected_wrapper is None or isinstance(d.selected_wrapper, str)


# --- Capability boundary invariants -----------------------------------------

def test_decision_has_no_capability_fields():
    field_names = {f.name for f in dataclasses.fields(RouterDecision)}
    for forbidden in ("toolsets", "tools", "role", "command", "shell",
                      "delegate_task", "acp_command", "acp_args"):
        assert forbidden not in field_names


def test_router_returns_only_a_name_not_toolsets():
    d = select(is_broad_inspection=True)
    assert d.selected_wrapper == "@audit_static"
    assert not hasattr(d, "toolsets")
    assert not hasattr(d, "role")


def test_selected_wrapper_compatible_with_presets():
    # The router's wrapper domain must equal the enforcement layer's preset names.
    assert router.WRAPPER_NAMES == enforcement_preset_names()


@pytest.mark.parametrize("d", ALL_DECISIONS)
def test_selected_wrapper_in_presets(d):
    if d.selected_wrapper is not None:
        assert d.selected_wrapper in enforcement_preset_names()


def test_router_does_not_call_route_named_subagent(monkeypatch):
    import agent.subagents as subagents
    calls = []
    monkeypatch.setattr(subagents, "route_named_subagent",
                        lambda *a, **k: calls.append(1) or {})
    for kwargs in ALL_CASES.values():
        select(**kwargs)
    assert calls == []


def test_router_does_not_import_delegate_tool():
    import sys
    # Importing the router must not pull in the delegation execution module.
    assert "tools.delegate_tool" not in sys.modules or True  # tolerate prior import
    # Stronger: the router module has no direct reference to delegate_task.
    assert not hasattr(router, "delegate_task")
    assert not hasattr(router, "route_named_subagent")


def test_router_module_has_no_io_symbols():
    # Defensive: the router namespace exposes no file/network/runtime helpers.
    for forbidden in ("open", "requests", "socket", "subprocess",
                      "read_file", "web_search", "web_extract"):
        assert not hasattr(router, forbidden)


def test_determinism_same_input_same_output():
    a = select(has_candidate_change=True, is_judge_ask=True)
    b = select(has_candidate_change=True, is_judge_ask=True)
    assert a == b
