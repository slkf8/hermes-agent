"""Offline tests for the confirmation policy module (M13).

Pure unit + contract tests over the explicit, auditable, non-NL safety
confirmation model. They assert the data-model shape/immutability, the import
boundary, source/assertion validation, the T1/T2/T3 tier permission matrix,
T3 non-clearability, SafePosture resolution, the danger-dominant
``merge_safety_posture``, audit inertness, determinism, and router round-trip
behavior. They use the normalizer + Central Router only as a round-trip oracle.
No runtime, no network, no file I/O, no delegate_task, no route_named_subagent.
"""

import ast
import dataclasses
import inspect

import pytest

from agent import subagent_confirmation_policy as cp
from agent.subagent_confirmation_policy import (
    AuditEffect,
    AuditEntry,
    AuditTrail,
    ConfirmationAssertion,
    ConfirmationSource,
    SafePosture,
    merge_safety_posture,
    resolve_dimension,
    validate_assertion,
)
from agent.subagent_task_descriptor import (
    RequestRef,
    RequestSignals,
    SignalValue,
    build_task_descriptor_from_signals,
)
from agent.subagent_router import select_named_subagent_for_task


# --- field vocabulary (kept local; partition is asserted against the module) -

SAFETY_FIELDS = (
    "mutation_requested",
    "execution_requested",
    "runtime_start_requested",
    "real_delegate_requested",
    "secret_or_forbidden_path_referenced",
    "external_surface_requested",
    "external_connect_requested",
    "git_mutation_requested",
    "env_mutation_requested",
)
T1 = (
    "mutation_requested",
    "execution_requested",
    "runtime_start_requested",
    "git_mutation_requested",
    "env_mutation_requested",
)
T2 = (
    "real_delegate_requested",
    "external_surface_requested",
    "external_connect_requested",
)
T3 = ("secret_or_forbidden_path_referenced",)

# safety RequestSignals field -> TaskDescriptor flag (R0 inputs)
SAFETY_TO_DESCRIPTOR = {
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

ALL_SOURCES = (
    ConfirmationSource.CENTRAL_POLICY,
    ConfirmationSource.HUMAN_EXPLICIT,
    ConfirmationSource.PREFLIGHT_GATE,
    ConfirmationSource.STATIC_SECURITY_GATE,
)


def _clearing_posture(fields, source=ConfirmationSource.CENTRAL_POLICY):
    """A posture that clears (FALSE) each field in ``fields``."""
    return SafePosture(
        assertions=tuple(
            ConfirmationAssertion(
                dimension=f, value=SignalValue.FALSE, source=source,
                reason=f"clear {f}",
            )
            for f in fields
        )
    )


# --- §A data-model shape & immutability -------------------------------------

def test_confirmation_source_members_exact():
    assert {s.name for s in ConfirmationSource} == {
        "CENTRAL_POLICY", "HUMAN_EXPLICIT", "PREFLIGHT_GATE", "STATIC_SECURITY_GATE",
    }
    assert len(list(ConfirmationSource)) == 4


def test_confirmation_source_no_nl_model_runtime():
    for s in ConfirmationSource:
        token = (s.name + s.value).upper()
        assert "NL" not in s.name
        for bad in ("MODEL", "LLM", "RUNTIME", "GATEWAY"):
            assert bad not in token


@pytest.mark.parametrize("klass", [ConfirmationAssertion, SafePosture, AuditEntry, AuditTrail])
def test_dataclasses_are_frozen(klass):
    assert dataclasses.fields(klass) is not None
    params = getattr(klass, "__dataclass_params__")
    assert params.frozen is True


def test_assertion_is_immutable():
    a = ConfirmationAssertion(
        dimension="mutation_requested", value=SignalValue.FALSE,
        source=ConfirmationSource.CENTRAL_POLICY, reason="r",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.reason = "other"  # type: ignore[misc]


def test_safeposture_default_is_empty_tuple():
    p = SafePosture()
    assert p.assertions == ()
    assert isinstance(p.assertions, tuple)


def test_no_timestamp_fields_anywhere():
    for klass in (ConfirmationAssertion, SafePosture, AuditEntry, AuditTrail):
        names = {f.name for f in dataclasses.fields(klass)}
        for temporal in ("timestamp", "ts", "time", "created_at", "expires_at", "when"):
            assert temporal not in names


# --- §B import allow / deny --------------------------------------------------

def _imported_modules_and_names():
    tree = ast.parse(inspect.getsource(cp))
    modules, names = set(), set()
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
    allowed = {"__future__", "dataclasses", "enum", "typing",
               "agent.subagent_task_descriptor"}
    assert modules <= allowed, f"unexpected imports: {modules - allowed}"


def test_no_forbidden_import_modules():
    modules, _ = _imported_modules_and_names()
    forbidden = {
        "agent.subagent_router", "agent.subagent_request_extractor",
        "agent.subagents", "tools.delegate_tool", "toolsets",
        "subprocess", "socket", "requests", "httpx", "pathlib", "os", "sys",
        "io", "time", "datetime", "random",
    }
    assert modules.isdisjoint(forbidden)


def test_no_forbidden_import_names():
    _, names = _imported_modules_and_names()
    forbidden = {
        "route_named_subagent", "delegate_task", "select_named_subagent_for_task",
        "TaskDescriptor", "RouterDecision", "build_task_descriptor_from_signals",
    }
    assert names.isdisjoint(forbidden)


def test_module_namespace_clean():
    for forbidden in (
        "open", "os", "sys", "pathlib", "subprocess", "socket", "requests",
        "httpx", "time", "datetime", "random",
        "route_named_subagent", "delegate_task", "select_named_subagent_for_task",
        "TaskDescriptor", "RouterDecision",
    ):
        assert not hasattr(cp, forbidden)


# --- §C / §D source & assertion validation -----------------------------------

@pytest.mark.parametrize("source", ALL_SOURCES)
def test_sources_constructible(source):
    # TRUE may be raised by any source on any field -> always valid.
    a = ConfirmationAssertion(
        dimension="mutation_requested", value=SignalValue.TRUE,
        source=source, reason="raise",
    )
    assert a.source is source


def test_invalid_dimension_raises():
    with pytest.raises(ValueError):
        ConfirmationAssertion(
            dimension="not_a_field", value=SignalValue.TRUE,
            source=ConfirmationSource.CENTRAL_POLICY, reason="r",
        )
    # an intent field name is also not a safety dimension
    with pytest.raises(ValueError):
        ConfirmationAssertion(
            dimension="judge_requested", value=SignalValue.FALSE,
            source=ConfirmationSource.CENTRAL_POLICY, reason="r",
        )


@pytest.mark.parametrize("reason", ["", "   ", "\t\n"])
def test_empty_reason_raises(reason):
    with pytest.raises(ValueError):
        ConfirmationAssertion(
            dimension="mutation_requested", value=SignalValue.FALSE,
            source=ConfirmationSource.CENTRAL_POLICY, reason=reason,
        )


def test_non_signalvalue_raises():
    with pytest.raises(ValueError):
        ConfirmationAssertion(
            dimension="mutation_requested", value="false",  # type: ignore[arg-type]
            source=ConfirmationSource.CENTRAL_POLICY, reason="r",
        )


def test_non_source_raises():
    with pytest.raises(ValueError):
        ConfirmationAssertion(
            dimension="mutation_requested", value=SignalValue.FALSE,
            source="central_policy",  # type: ignore[arg-type]
            reason="r",
        )


def test_validate_assertion_helper_matches_constructor():
    # the standalone validator accepts a permitted clear ...
    validate_assertion(
        "mutation_requested", SignalValue.FALSE,
        ConfirmationSource.CENTRAL_POLICY, "ok",
    )
    # ... and rejects a disallowed one loudly.
    with pytest.raises(ValueError):
        validate_assertion(
            "secret_or_forbidden_path_referenced", SignalValue.FALSE,
            ConfirmationSource.CENTRAL_POLICY, "nope",
        )


# --- §E per-field & partition ------------------------------------------------

def test_module_safety_fields_match_local_and_request_signals():
    assert cp._SAFETY_FIELDS == SAFETY_FIELDS
    rs_fields = {f.name for f in dataclasses.fields(RequestSignals)}
    for f in SAFETY_FIELDS:
        assert f in rs_fields


def test_tier_partition_is_exact_and_disjoint():
    assert cp._T1_FIELDS == frozenset(T1)
    assert cp._T2_FIELDS == frozenset(T2)
    assert cp._T3_FIELDS == frozenset(T3)
    assert cp._T1_FIELDS | cp._T2_FIELDS | cp._T3_FIELDS == set(SAFETY_FIELDS)
    assert cp._T1_FIELDS.isdisjoint(cp._T2_FIELDS)
    assert cp._T1_FIELDS.isdisjoint(cp._T3_FIELDS)
    assert cp._T2_FIELDS.isdisjoint(cp._T3_FIELDS)


@pytest.mark.parametrize("field", SAFETY_FIELDS)
def test_field_default_unknown_maps_true(field):
    merged, _audit = merge_safety_posture(RequestSignals(), SafePosture())
    assert getattr(merged, field) is SignalValue.UNKNOWN
    td = build_task_descriptor_from_signals(merged)
    assert getattr(td, SAFETY_TO_DESCRIPTOR[field]) is True


@pytest.mark.parametrize("field", SAFETY_FIELDS)
def test_true_raise_allowed_any_source_any_field(field):
    for source in ALL_SOURCES:
        a = ConfirmationAssertion(
            dimension=field, value=SignalValue.TRUE, source=source, reason="raise",
        )
        assert a.value is SignalValue.TRUE


# --- §F tier permission matrix ----------------------------------------------

@pytest.mark.parametrize("field", T1)
def test_t1_central_policy_false_allowed(field):
    a = ConfirmationAssertion(
        dimension=field, value=SignalValue.FALSE,
        source=ConfirmationSource.CENTRAL_POLICY, reason="read-only profile",
    )
    assert a.value is SignalValue.FALSE


@pytest.mark.parametrize("field", T1)
def test_t1_human_explicit_false_allowed(field):
    a = ConfirmationAssertion(
        dimension=field, value=SignalValue.FALSE,
        source=ConfirmationSource.HUMAN_EXPLICIT, reason="human confirmed",
    )
    assert a.value is SignalValue.FALSE


@pytest.mark.parametrize("field", T2)
def test_t2_central_policy_false_allowed(field):
    a = ConfirmationAssertion(
        dimension=field, value=SignalValue.FALSE,
        source=ConfirmationSource.CENTRAL_POLICY, reason="leaf wrapper invariant",
    )
    assert a.value is SignalValue.FALSE


@pytest.mark.parametrize("field", T2)
def test_t2_human_explicit_false_rejected(field):
    with pytest.raises(ValueError):
        ConfirmationAssertion(
            dimension=field, value=SignalValue.FALSE,
            source=ConfirmationSource.HUMAN_EXPLICIT, reason="should fail",
        )


@pytest.mark.parametrize("field", SAFETY_FIELDS)
def test_gate_sources_clear_nothing(field):
    for source in (ConfirmationSource.PREFLIGHT_GATE,
                   ConfirmationSource.STATIC_SECURITY_GATE):
        with pytest.raises(ValueError):
            ConfirmationAssertion(
                dimension=field, value=SignalValue.FALSE, source=source,
                reason="gates are inert in MVP",
            )


# --- §I T3 non-clearability --------------------------------------------------

@pytest.mark.parametrize("source", ALL_SOURCES)
def test_t3_false_rejected_all_sources(source):
    with pytest.raises(ValueError):
        ConfirmationAssertion(
            dimension="secret_or_forbidden_path_referenced",
            value=SignalValue.FALSE, source=source, reason="cannot clear T3",
        )


def test_t3_true_dominates_through_merge():
    nl = RequestSignals(secret_or_forbidden_path_referenced=SignalValue.TRUE)
    merged, _audit = merge_safety_posture(nl, SafePosture())
    assert merged.secret_or_forbidden_path_referenced is SignalValue.TRUE
    td = build_task_descriptor_from_signals(merged)
    assert td.targets_secret_or_forbidden_path is True


# --- §G SafePosture resolution ----------------------------------------------

def test_empty_posture_all_unknown():
    p = SafePosture()
    for field in SAFETY_FIELDS:
        assert resolve_dimension(p, field) is SignalValue.UNKNOWN


def test_true_dominates_false_within_posture():
    p = SafePosture(assertions=(
        ConfirmationAssertion("mutation_requested", SignalValue.FALSE,
                              ConfirmationSource.CENTRAL_POLICY, "clear"),
        ConfirmationAssertion("mutation_requested", SignalValue.TRUE,
                              ConfirmationSource.CENTRAL_POLICY, "raise"),
    ))
    assert resolve_dimension(p, "mutation_requested") is SignalValue.TRUE


def test_permitted_false_resolves_false():
    p = _clearing_posture(("mutation_requested",))
    assert resolve_dimension(p, "mutation_requested") is SignalValue.FALSE


def test_no_cross_field_inference():
    p = _clearing_posture(("mutation_requested",))
    assert resolve_dimension(p, "mutation_requested") is SignalValue.FALSE
    for field in SAFETY_FIELDS:
        if field != "mutation_requested":
            assert resolve_dimension(p, field) is SignalValue.UNKNOWN


def test_no_blanket_all_safe_api():
    # there is no convenience constructor/function that clears everything at once
    for name in ("all_safe", "clear_all", "blanket", "mark_all_safe", "trust_all"):
        assert not hasattr(cp, name)
    # an empty posture clears nothing
    p = SafePosture()
    for field in SAFETY_FIELDS:
        assert resolve_dimension(p, field) is SignalValue.UNKNOWN


def test_resolve_dimension_rejects_unknown_field():
    with pytest.raises(ValueError):
        resolve_dimension(SafePosture(), "not_a_field")


# --- §H merge core -----------------------------------------------------------

def test_merge_nl_true_plus_conf_false_is_true():
    nl = RequestSignals(mutation_requested=SignalValue.TRUE)
    posture = _clearing_posture(("mutation_requested",))
    merged, _audit = merge_safety_posture(nl, posture)
    assert merged.mutation_requested is SignalValue.TRUE  # NL TRUE dominates


def test_merge_nl_unknown_plus_conf_false_is_false():
    nl = RequestSignals()
    posture = _clearing_posture(("mutation_requested",))
    merged, _audit = merge_safety_posture(nl, posture)
    assert merged.mutation_requested is SignalValue.FALSE


def test_merge_nl_unknown_plus_conf_true_is_true():
    nl = RequestSignals()
    posture = SafePosture(assertions=(
        ConfirmationAssertion("mutation_requested", SignalValue.TRUE,
                              ConfirmationSource.CENTRAL_POLICY, "raise"),
    ))
    merged, _audit = merge_safety_posture(nl, posture)
    assert merged.mutation_requested is SignalValue.TRUE


def test_merge_nl_unknown_plus_posture_unknown_is_unknown():
    merged, _audit = merge_safety_posture(RequestSignals(), SafePosture())
    for field in SAFETY_FIELDS:
        assert getattr(merged, field) is SignalValue.UNKNOWN


def test_merge_intent_fields_unchanged():
    nl = RequestSignals(
        public_url_present=SignalValue.TRUE,
        judge_requested=SignalValue.TRUE,
        gate_question_present=SignalValue.TRUE,
    )
    merged, _audit = merge_safety_posture(nl, _clearing_posture(T1 + T2))
    for f in (
        "public_url_present", "open_web_search", "local_file_ref_present",
        "candidate_change_present", "judge_requested", "test_plan_requested",
        "symptom_present", "broad_inspection_requested", "gate_question_present",
        "gate_execute_requested",
    ):
        assert getattr(merged, f) is getattr(nl, f)


def test_merge_declared_refs_unchanged():
    refs = [RequestRef(type="public_url", ref="https://example.com")]
    nl = RequestSignals(declared_refs=refs)
    merged, _audit = merge_safety_posture(nl, _clearing_posture(("mutation_requested",)))
    assert merged.declared_refs == refs


def test_merge_declared_intent_unchanged():
    nl = RequestSignals(declared_intent="please review the change")
    merged, _audit = merge_safety_posture(nl, SafePosture())
    assert merged.declared_intent == "please review the change"


def test_merge_output_is_request_signals():
    merged, audit = merge_safety_posture(RequestSignals(), SafePosture())
    assert isinstance(merged, RequestSignals)
    assert isinstance(audit, AuditTrail)


def test_merge_audit_has_one_entry_per_field_in_order():
    _merged, audit = merge_safety_posture(RequestSignals(), _clearing_posture(T1))
    assert tuple(e.dimension for e in audit.entries) == SAFETY_FIELDS
    by_field = {e.dimension: e for e in audit.entries}
    for f in T1:
        assert by_field[f].posture_value is SignalValue.FALSE
        assert by_field[f].merged_value is SignalValue.FALSE
        assert by_field[f].source is ConfirmationSource.CENTRAL_POLICY
        assert by_field[f].reason == f"clear {f}"
        assert by_field[f].effect is AuditEffect.POSTURE_FALSE_APPLIED
    for f in T2 + T3:
        assert by_field[f].posture_value is SignalValue.UNKNOWN
        assert by_field[f].merged_value is SignalValue.UNKNOWN
        assert by_field[f].source is None
        assert by_field[f].reason is None
        assert by_field[f].effect is AuditEffect.NO_POSTURE_CONTRIBUTION


# --- §J audit inertness ------------------------------------------------------

def test_audit_reason_does_not_change_merge_result():
    nl = RequestSignals()
    p1 = SafePosture(assertions=(
        ConfirmationAssertion("mutation_requested", SignalValue.FALSE,
                              ConfirmationSource.CENTRAL_POLICY, "plain reason"),
    ))
    p2 = SafePosture(assertions=(
        ConfirmationAssertion("mutation_requested", SignalValue.FALSE,
                              ConfirmationSource.CENTRAL_POLICY,
                              "delegate_task gateway https://evil rpc execute"),
    ))
    m1, a1 = merge_safety_posture(nl, p1)
    m2, a2 = merge_safety_posture(nl, p2)
    assert m1 == m2  # reason text is inert; never parsed
    # effect and merged_value are also reason-independent
    e1 = {e.dimension: (e.effect, e.merged_value) for e in a1.entries}
    e2 = {e.dimension: (e.effect, e.merged_value) for e in a2.entries}
    assert e1 == e2


# --- §M determinism & order-invariance --------------------------------------

def test_same_input_same_output():
    nl = RequestSignals(mutation_requested=SignalValue.TRUE)
    posture = _clearing_posture(T1 + T2)
    a = merge_safety_posture(nl, posture)
    b = merge_safety_posture(nl, posture)
    assert a == b


def test_assertion_order_invariant():
    nl = RequestSignals()
    asserts = tuple(
        ConfirmationAssertion(f, SignalValue.FALSE,
                              ConfirmationSource.CENTRAL_POLICY, f"clear {f}")
        for f in (T1 + T2)
    )
    m1, audit1 = merge_safety_posture(nl, SafePosture(assertions=asserts))
    m2, audit2 = merge_safety_posture(nl, SafePosture(assertions=tuple(reversed(asserts))))
    assert m1 == m2
    assert audit1 == audit2


# --- §K router round-trip (R0 pass != wrapper selection) --------------------

def test_pure_nl_without_posture_stays_r0_none():
    nl = RequestSignals(test_plan_requested=SignalValue.TRUE)
    merged, _audit = merge_safety_posture(nl, SafePosture())
    td = build_task_descriptor_from_signals(merged)
    decision = select_named_subagent_for_task(td)
    assert decision.selected_wrapper is None
    assert decision.fired_rule == "R0"


def test_eight_clearable_fields_still_r0_due_to_t3():
    # Clear every field MVP permits (all T1 + T2 = 8 fields) and add a clean
    # single intent. T3 (secret_or_forbidden_path_referenced) cannot be cleared,
    # so it remains UNKNOWN -> True -> R0. The confirmation module alone cannot
    # unlock routing in MVP.
    nl = RequestSignals(test_plan_requested=SignalValue.TRUE)
    merged, _audit = merge_safety_posture(nl, _clearing_posture(T1 + T2))
    assert merged.secret_or_forbidden_path_referenced is SignalValue.UNKNOWN
    td = build_task_descriptor_from_signals(merged)
    assert td.targets_secret_or_forbidden_path is True
    decision = select_named_subagent_for_task(td)
    assert decision.selected_wrapper is None
    assert decision.fired_rule == "R0"


def test_confirmation_never_emits_wrapper_or_decision():
    # merge_safety_posture returns only (RequestSignals, AuditTrail); it never
    # yields a wrapper name or RouterDecision.
    merged, audit = merge_safety_posture(RequestSignals(), _clearing_posture(T1))
    assert type(merged).__name__ == "RequestSignals"
    assert type(audit).__name__ == "AuditTrail"
    assert not hasattr(merged, "selected_wrapper")


# --- helpers for audit groups -----------------------------------------------

def _entry_for(audit, dimension):
    by_field = {e.dimension: e for e in audit.entries}
    return by_field[dimension]


def _one_assertion_posture(dimension, value, source=ConfirmationSource.CENTRAL_POLICY,
                           reason="r"):
    return SafePosture(assertions=(
        ConfirmationAssertion(dimension, value, source, reason),
    ))


# --- Group A: AuditEffect enum exactness ------------------------------------

def test_audit_effect_members_exact():
    assert {e.name for e in AuditEffect} == {
        "POSTURE_FALSE_OVERRIDDEN_BY_NL_TRUE",
        "NL_TRUE_PRESERVED",
        "POSTURE_TRUE_RAISED",
        "POSTURE_FALSE_APPLIED",
        "POSTURE_UNKNOWN_INERT",
        "NO_POSTURE_CONTRIBUTION",
    }
    assert len(list(AuditEffect)) == 6


# --- Group B: AuditEntry schema / field exposure ----------------------------

def test_audit_entry_fields_exposed():
    names = {f.name for f in dataclasses.fields(AuditEntry)}
    assert names == {
        "dimension", "nl_value", "posture_value", "merged_value",
        "source", "reason", "effect", "overridden_by_nl_true",
    }


def test_audit_entry_no_timestamp_field():
    names = {f.name for f in dataclasses.fields(AuditEntry)}
    for temporal in ("timestamp", "ts", "time", "created_at", "expires_at", "when"):
        assert temporal not in names


# --- Group C: effect classifier cases ---------------------------------------

def test_effect_nl_true_posture_false_overridden():
    nl = RequestSignals(mutation_requested=SignalValue.TRUE)
    merged, audit = merge_safety_posture(nl, _clearing_posture(("mutation_requested",)))
    e = _entry_for(audit, "mutation_requested")
    assert merged.mutation_requested is SignalValue.TRUE
    assert e.effect is AuditEffect.POSTURE_FALSE_OVERRIDDEN_BY_NL_TRUE
    assert e.overridden_by_nl_true is True
    assert e.nl_value is SignalValue.TRUE
    assert e.posture_value is SignalValue.FALSE
    assert e.merged_value is SignalValue.TRUE


def test_effect_nl_unknown_posture_false_applied():
    merged, audit = merge_safety_posture(
        RequestSignals(), _clearing_posture(("mutation_requested",)))
    e = _entry_for(audit, "mutation_requested")
    assert merged.mutation_requested is SignalValue.FALSE
    assert e.effect is AuditEffect.POSTURE_FALSE_APPLIED
    assert e.overridden_by_nl_true is False


def test_effect_nl_unknown_posture_true_raised():
    posture = _one_assertion_posture("mutation_requested", SignalValue.TRUE,
                                     reason="raise")
    merged, audit = merge_safety_posture(RequestSignals(), posture)
    e = _entry_for(audit, "mutation_requested")
    assert merged.mutation_requested is SignalValue.TRUE
    assert e.effect is AuditEffect.POSTURE_TRUE_RAISED


def test_effect_nl_true_posture_unknown_preserved():
    nl = RequestSignals(mutation_requested=SignalValue.TRUE)
    merged, audit = merge_safety_posture(nl, SafePosture())
    e = _entry_for(audit, "mutation_requested")
    assert merged.mutation_requested is SignalValue.TRUE
    assert e.effect is AuditEffect.NL_TRUE_PRESERVED
    assert e.overridden_by_nl_true is False


def test_effect_no_posture_contribution():
    _merged, audit = merge_safety_posture(RequestSignals(), SafePosture())
    e = _entry_for(audit, "mutation_requested")
    assert e.effect is AuditEffect.NO_POSTURE_CONTRIBUTION
    assert e.source is None
    assert e.merged_value is SignalValue.UNKNOWN


def test_effect_posture_unknown_inert():
    posture = _one_assertion_posture("mutation_requested", SignalValue.UNKNOWN,
                                     reason="inert")
    merged, audit = merge_safety_posture(RequestSignals(), posture)
    e = _entry_for(audit, "mutation_requested")
    assert merged.mutation_requested is SignalValue.UNKNOWN
    assert e.effect is AuditEffect.POSTURE_UNKNOWN_INERT


def test_effect_nl_true_posture_true_labeled_nl_true_preserved():
    nl = RequestSignals(mutation_requested=SignalValue.TRUE)
    posture = _one_assertion_posture("mutation_requested", SignalValue.TRUE,
                                     reason="raise")
    merged, audit = merge_safety_posture(nl, posture)
    e = _entry_for(audit, "mutation_requested")
    assert merged.mutation_requested is SignalValue.TRUE
    assert e.effect is AuditEffect.NL_TRUE_PRESERVED  # deterministic tie-break
    assert e.posture_value is SignalValue.TRUE  # posture raise still recorded


# --- Group D: merged_value cross-consistency --------------------------------

def test_audit_merged_value_matches_request_signals():
    nl = RequestSignals(
        mutation_requested=SignalValue.TRUE,
        real_delegate_requested=SignalValue.TRUE,
    )
    merged, audit = merge_safety_posture(nl, _clearing_posture(T1 + T2))
    for e in audit.entries:
        assert e.merged_value is getattr(merged, e.dimension)


# --- Group H additions: clear-map coverage ----------------------------------

def test_every_source_keyed_in_clear_map():
    assert set(cp._FALSE_CLEAR_BY_SOURCE) == set(ConfirmationSource)


def test_gate_source_clear_sets_empty():
    assert cp._FALSE_CLEAR_BY_SOURCE[ConfirmationSource.PREFLIGHT_GATE] == frozenset()
    assert cp._FALSE_CLEAR_BY_SOURCE[ConfirmationSource.STATIC_SECURITY_GATE] == frozenset()


def test_safety_fields_match_request_signals_declaration_order():
    safety = {
        "mutation_requested", "execution_requested", "runtime_start_requested",
        "real_delegate_requested", "secret_or_forbidden_path_referenced",
        "external_surface_requested", "external_connect_requested",
        "git_mutation_requested", "env_mutation_requested",
    }
    ordered = tuple(
        f.name for f in dataclasses.fields(RequestSignals) if f.name in safety
    )
    assert cp._SAFETY_FIELDS == ordered
