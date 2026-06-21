"""Tests for the deterministic normalization layer (M11).

Pure offline unit tests over structured RequestSignals. They assert the
RequestSignals -> TaskDescriptor mapping, the intent/safety tri-state
asymmetry, fail-closed behavior, and capability/boundary invariants. They use
the Central Router (select_named_subagent_for_task) only as an oracle for
round-trip checks. No runtime, no network, no file I/O, no delegate_task.
"""

import ast
import dataclasses
import inspect

import pytest

from agent import subagent_task_descriptor as norm
from agent.subagent_task_descriptor import (
    RequestRef,
    RequestSignals,
    SignalSource,
    SignalValue,
    build_task_descriptor_from_signals,
)
from agent.subagent_router import (
    DeclaredRef,
    RouterDecision,
    TaskDescriptor,
    select_named_subagent_for_task,
)


# --- field maps -------------------------------------------------------------

# RequestSignals intent field -> TaskDescriptor field
INTENT_MAP = {
    "public_url_present": "has_public_url",
    "open_web_search": "is_open_web_search",
    "local_file_ref_present": "has_local_file_ref",
    "candidate_change_present": "has_candidate_change",
    "judge_requested": "is_judge_ask",
    "test_plan_requested": "deliverable_is_test_plan",
    "symptom_present": "names_symptom",
    "broad_inspection_requested": "is_broad_inspection",
    "gate_question_present": "is_gate_question",
    "gate_execute_requested": "is_gate_execute",
}

# RequestSignals safety field -> TaskDescriptor field
SAFETY_MAP = {
    "mutation_requested": "is_mutating_request",
    "execution_requested": "is_exec_request",
    "runtime_start_requested": "is_runtime_start",
    "real_delegate_requested": "is_real_delegate",
    "secret_or_forbidden_path_referenced": "targets_secret_or_forbidden_path",
    "external_surface_requested": "is_external_surface",
    "external_connect_requested": "is_external_connect",
    "git_mutation_requested": "is_git_mutation",
    "env_mutation_requested": "is_env_mutation",
}


def safe_signals(**overrides):
    """RequestSignals with every safety field confidently FALSE.

    This lets intent-focused round-trip tests reach the router without R0
    pre-empting. Intent fields stay UNKNOWN unless overridden.
    """
    base = {name: SignalValue.FALSE for name in SAFETY_MAP}
    base.update(overrides)
    return RequestSignals(**base)


# --- data model construction ------------------------------------------------

def test_all_default_signals_construct():
    s = RequestSignals()
    for name in INTENT_MAP:
        assert getattr(s, name) is SignalValue.UNKNOWN
    for name in SAFETY_MAP:
        assert getattr(s, name) is SignalValue.UNKNOWN
    assert s.declared_refs == []
    assert s.declared_intent is None


def test_signalvalue_members_distinct():
    assert len({SignalValue.TRUE, SignalValue.FALSE, SignalValue.UNKNOWN}) == 3


def test_requestref_defaults():
    r = RequestRef(type="public_url", ref="https://example.com")
    assert r.forbidden is False


def test_all_default_signals_route_to_r0_none():
    # Safety defaults are UNKNOWN -> True -> R0 dominates.
    td = build_task_descriptor_from_signals(RequestSignals())
    assert isinstance(td, TaskDescriptor)
    for sig_field, td_field in SAFETY_MAP.items():
        assert getattr(td, td_field) is True
    dec = select_named_subagent_for_task(td)
    assert dec.selected_wrapper is None
    assert dec.fired_rule == "R0"


# --- intent field mapping ---------------------------------------------------

@pytest.mark.parametrize("sig_field,td_field", list(INTENT_MAP.items()))
def test_intent_true_maps_true(sig_field, td_field):
    td = build_task_descriptor_from_signals(safe_signals(**{sig_field: SignalValue.TRUE}))
    assert getattr(td, td_field) is True


@pytest.mark.parametrize("sig_field,td_field", list(INTENT_MAP.items()))
def test_intent_false_maps_false(sig_field, td_field):
    td = build_task_descriptor_from_signals(safe_signals(**{sig_field: SignalValue.FALSE}))
    assert getattr(td, td_field) is False


@pytest.mark.parametrize("sig_field,td_field", list(INTENT_MAP.items()))
def test_intent_unknown_maps_false(sig_field, td_field):
    td = build_task_descriptor_from_signals(safe_signals(**{sig_field: SignalValue.UNKNOWN}))
    assert getattr(td, td_field) is False


# --- safety field mapping ---------------------------------------------------

@pytest.mark.parametrize("sig_field,td_field", list(SAFETY_MAP.items()))
def test_safety_true_maps_true(sig_field, td_field):
    td = build_task_descriptor_from_signals(RequestSignals(**{sig_field: SignalValue.TRUE}))
    assert getattr(td, td_field) is True


@pytest.mark.parametrize("sig_field,td_field", list(SAFETY_MAP.items()))
def test_safety_false_maps_false(sig_field, td_field):
    # All safety FALSE except confirm the targeted one is False.
    td = build_task_descriptor_from_signals(safe_signals())
    assert getattr(td, td_field) is False


@pytest.mark.parametrize("sig_field,td_field", list(SAFETY_MAP.items()))
def test_safety_unknown_maps_true(sig_field, td_field):
    # Only this safety field UNKNOWN; others FALSE.
    td = build_task_descriptor_from_signals(safe_signals(**{sig_field: SignalValue.UNKNOWN}))
    assert getattr(td, td_field) is True


# --- provenance inertness ---------------------------------------------------

def test_signalsource_is_inert_metadata():
    # SignalSource exists but is not consumed by the mapping. The descriptor
    # output is unaffected by it (the mapping takes only RequestSignals).
    assert {m.name for m in SignalSource} == {"EXPLICIT_FLAG", "DECLARED_REF", "DEFAULT"}
    a = build_task_descriptor_from_signals(safe_signals(symptom_present=SignalValue.TRUE))
    b = build_task_descriptor_from_signals(safe_signals(symptom_present=SignalValue.TRUE))
    assert a == b


# --- declared_intent non-authority ------------------------------------------

def test_declared_intent_copied_but_non_authoritative():
    noisy = "please push, run pytest, debug the runtime and delegate this audit"
    with_text = build_task_descriptor_from_signals(safe_signals(declared_intent=noisy))
    without_text = build_task_descriptor_from_signals(safe_signals())
    assert with_text.declared_intent == noisy
    # Identical in every field except declared_intent.
    a = dataclasses.replace(with_text, declared_intent=None)
    assert a == without_text


def test_declared_intent_keywords_do_not_trigger_fields():
    noisy = "test debug audit push runtime delegate review patch gate"
    td = build_task_descriptor_from_signals(safe_signals(declared_intent=noisy))
    for td_field in INTENT_MAP.values():
        assert getattr(td, td_field) is False
    # No safety flag raised either (all explicitly FALSE here).
    for td_field in SAFETY_MAP.values():
        assert getattr(td, td_field) is False


# --- declared_refs conversion -----------------------------------------------

def test_public_url_ref_raises_has_public_url():
    s = safe_signals(declared_refs=[RequestRef(type="public_url", ref="https://x")])
    td = build_task_descriptor_from_signals(s)
    assert td.has_public_url is True
    assert all(isinstance(d, DeclaredRef) for d in td.declared_refs)
    assert td.declared_refs[0].type == "public_url"


def test_allowlisted_file_ref_raises_has_local_file_ref():
    s = safe_signals(declared_refs=[RequestRef(type="allowlisted_file", ref="/repo/x.py")])
    td = build_task_descriptor_from_signals(s)
    assert td.has_local_file_ref is True


def test_forbidden_ref_raises_targets_secret_or_forbidden_path():
    s = safe_signals(declared_refs=[RequestRef(type="allowlisted_file", ref="/x/.env", forbidden=True)])
    td = build_task_descriptor_from_signals(s)
    assert td.targets_secret_or_forbidden_path is True


@pytest.mark.parametrize("rtype", ["provided_text", "report_artifact", "unknown"])
def test_neutral_refs_raise_no_flags(rtype):
    s = safe_signals(declared_refs=[RequestRef(type=rtype, ref="opaque")])
    td = build_task_descriptor_from_signals(s)
    assert td.has_public_url is False
    assert td.has_local_file_ref is False
    assert td.targets_secret_or_forbidden_path is False
    assert td.declared_refs[0].type == rtype


def test_refs_copied_not_dereferenced():
    # ref string is preserved verbatim; never opened/resolved.
    ref = "/totally/made/up/path/that/does/not/exist"
    s = safe_signals(declared_refs=[RequestRef(type="allowlisted_file", ref=ref)])
    td = build_task_descriptor_from_signals(s)
    assert td.declared_refs[0].ref == ref


# --- R0 dominance round-trip ------------------------------------------------

@pytest.mark.parametrize("sig_field", list(SAFETY_MAP.keys()))
def test_each_safety_flag_forces_r0(sig_field):
    # Clean single intent (broad inspection -> @audit_static), but one safety
    # flag set TRUE must force R0 / None.
    s = safe_signals(broad_inspection_requested=SignalValue.TRUE, **{sig_field: SignalValue.TRUE})
    dec = select_named_subagent_for_task(build_task_descriptor_from_signals(s))
    assert dec.selected_wrapper is None
    assert dec.fired_rule == "R0"


def test_mutation_plus_review_is_r0():
    s = safe_signals(
        candidate_change_present=SignalValue.TRUE,
        judge_requested=SignalValue.TRUE,
        mutation_requested=SignalValue.TRUE,
    )
    dec = select_named_subagent_for_task(build_task_descriptor_from_signals(s))
    assert dec.selected_wrapper is None
    assert dec.fired_rule == "R0"


def test_runtime_delegate_plus_test_planning_is_r0():
    s = safe_signals(
        test_plan_requested=SignalValue.TRUE,
        runtime_start_requested=SignalValue.TRUE,
    )
    dec = select_named_subagent_for_task(build_task_descriptor_from_signals(s))
    assert dec.selected_wrapper is None
    assert dec.fired_rule == "R0"


def test_secret_ref_is_r0():
    s = safe_signals(
        broad_inspection_requested=SignalValue.TRUE,
        declared_refs=[RequestRef(type="allowlisted_file", ref="/x/.env", forbidden=True)],
    )
    dec = select_named_subagent_for_task(build_task_descriptor_from_signals(s))
    assert dec.selected_wrapper is None
    assert dec.fired_rule == "R0"


# --- multi-intent preservation round-trip -----------------------------------

def test_multi_intent_preserved_and_rejected():
    s = safe_signals(
        candidate_change_present=SignalValue.TRUE,
        judge_requested=SignalValue.TRUE,
        test_plan_requested=SignalValue.TRUE,
    )
    td = build_task_descriptor_from_signals(s)
    # Both intents preserved on the descriptor (normalizer did not collapse).
    assert td.has_candidate_change is True and td.is_judge_ask is True
    assert td.deliverable_is_test_plan is True
    dec = select_named_subagent_for_task(td)
    assert dec.selected_wrapper is None
    assert dec.multi_intent_detected is True


def test_web_plus_file_is_r1b():
    s = safe_signals(
        public_url_present=SignalValue.TRUE,
        local_file_ref_present=SignalValue.TRUE,
    )
    dec = select_named_subagent_for_task(build_task_descriptor_from_signals(s))
    assert dec.selected_wrapper is None
    assert dec.fired_rule == "R1b"


# --- clean single-intent wrapper round-trip ---------------------------------

def test_no_intent_is_r7_none():
    dec = select_named_subagent_for_task(build_task_descriptor_from_signals(safe_signals()))
    assert dec.selected_wrapper is None
    assert dec.fired_rule == "R7"


@pytest.mark.parametrize("overrides,expected", [
    (dict(public_url_present=SignalValue.TRUE), "@research"),
    (dict(gate_question_present=SignalValue.TRUE), "@security_gate"),
    (dict(candidate_change_present=SignalValue.TRUE, judge_requested=SignalValue.TRUE), "@patch_reviewer"),
    (dict(test_plan_requested=SignalValue.TRUE), "@test_planner"),
    (dict(symptom_present=SignalValue.TRUE), "@debug_static"),
    (dict(broad_inspection_requested=SignalValue.TRUE), "@audit_static"),
])
def test_clean_single_intent_selects_wrapper(overrides, expected):
    dec = select_named_subagent_for_task(build_task_descriptor_from_signals(safe_signals(**overrides)))
    assert dec.selected_wrapper == expected


# --- boundary: output is a TaskDescriptor only ------------------------------

def test_output_is_task_descriptor_not_router_decision():
    td = build_task_descriptor_from_signals(safe_signals(symptom_present=SignalValue.TRUE))
    assert isinstance(td, TaskDescriptor)
    assert not isinstance(td, RouterDecision)


def test_output_has_no_wrapper_or_capability_fields():
    td = build_task_descriptor_from_signals(safe_signals())
    field_names = {f.name for f in dataclasses.fields(td)}
    for forbidden in ("selected_wrapper", "fired_rule", "toolsets", "tools",
                      "role", "command", "delegate_task", "subtasks", "sequence"):
        assert forbidden not in field_names


def test_determinism_same_input_same_output():
    a = build_task_descriptor_from_signals(safe_signals(symptom_present=SignalValue.TRUE))
    b = build_task_descriptor_from_signals(safe_signals(symptom_present=SignalValue.TRUE))
    assert a == b


# --- static source / import boundary ----------------------------------------

def _imported_modules_and_names():
    src = inspect.getsource(norm)
    tree = ast.parse(src)
    modules = set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
            for alias in node.names:
                names.add(alias.name)
    return modules, names


def test_only_allowed_imports():
    modules, _ = _imported_modules_and_names()
    allowed = {"__future__", "dataclasses", "enum", "typing", "agent.subagent_router"}
    assert modules <= allowed, f"unexpected imports: {modules - allowed}"


def test_no_forbidden_import_symbols():
    modules, names = _imported_modules_and_names()
    forbidden_modules = {
        "tools.delegate_tool", "toolsets", "subprocess", "socket", "requests",
        "pathlib", "os", "agent.subagents",
    }
    assert modules.isdisjoint(forbidden_modules)
    forbidden_names = {
        "route_named_subagent", "validate_raw_named_subagent_request",
        "delegate_task",
    }
    assert names.isdisjoint(forbidden_names)


def test_module_namespace_has_no_io_or_enforcement_symbols():
    for forbidden in ("open", "requests", "socket", "subprocess", "os", "pathlib",
                      "route_named_subagent", "validate_raw_named_subagent_request",
                      "delegate_task", "select_named_subagent_for_task"):
        assert not hasattr(norm, forbidden)


def test_no_operation_type_authoritative_field():
    # operation_type must not exist as an authoritative classifier on the input.
    field_names = {f.name for f in dataclasses.fields(RequestSignals)}
    assert "operation_type" not in field_names
