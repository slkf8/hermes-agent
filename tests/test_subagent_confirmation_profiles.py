"""Offline tests for the confirmation profile producer module (M15).

Pure unit + contract tests over the sanctioned ``CENTRAL_POLICY`` posture
producer. They assert the import boundary, data-model shape/immutability,
producer validity, loud rejections (incl. T3 non-clearability and duplicate /
empty-clears rejection), test-only merge integration, test-only router safety
(T3 still forces R0), the documented source-spoofing caveat, the static
registry, and anti-regression invariants.

No runtime, no network, no file I/O by the production module, no delegate_task,
no route_named_subagent. The normalizer, router, and extractor are used only as
round-trip oracles.
"""

import ast
import dataclasses
import inspect

import pytest

from agent import subagent_confirmation_profiles as profiles
from agent.subagent_confirmation_profiles import (
    ConfirmationProfile,
    KNOWN_PROFILE_NAMES,
    ProfileClear,
    build_posture_by_name,
    build_posture_from_profile,
    validate_profile,
)
from agent import subagent_confirmation_policy as cp
from agent.subagent_confirmation_policy import (
    AuditEffect,
    ConfirmationAssertion,
    ConfirmationSource,
    SafePosture,
    merge_safety_posture,
    validate_assertion,
)
from agent.subagent_task_descriptor import (
    RequestSignals,
    SignalValue,
    build_task_descriptor_from_signals,
)
from agent.subagent_router import select_named_subagent_for_task
from agent.subagent_request_extractor import extract_request_signals_from_text


# --- tier vocabulary (kept local; mirrors the policy module) ----------------

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

ALLOWED_IMPORT_MODULES = {
    "__future__",
    "dataclasses",
    "types",
    "agent.subagent_confirmation_policy",
    "agent.subagent_task_descriptor",
}

FORBIDDEN_IMPORT_ROOTS = {
    "os",
    "pathlib",
    "io",
    "socket",
    "subprocess",
    "time",
    "datetime",
    "random",
    "secrets",
    "requests",
    "httpx",
    "agent",  # only specific agent.* modules are allowed (checked separately)
    "tools",
}

FORBIDDEN_QUALIFIED = {
    "agent.subagent_router",
    "agent.subagent_request_extractor",
    "agent.subagents",
    "tools.delegate_tool",
}


def _module_tree():
    return ast.parse(inspect.getsource(profiles))


def _imported_modules():
    """Return the set of fully-qualified module strings imported by the module."""
    mods = set()
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name)
    return mods


def _make_maximal_t1_t2_profile():
    """Build (in-test only) the most permissive valid profile: all T1+T2, no T3."""
    clears = tuple(
        ProfileClear(dimension=dim, reason=f"test clearance for {dim}")
        for dim in (T1 + T2)
    )
    return ConfirmationProfile(name="test_maximal_t1_t2", clears=clears)


# =====================================================================
# Group A -- module boundary / import safety
# =====================================================================

def test_A_allowed_import_surface_only():
    for mod in _imported_modules():
        assert mod in ALLOWED_IMPORT_MODULES, f"unexpected import: {mod}"


def test_A_forbidden_import_roots_absent():
    for mod in _imported_modules():
        root = mod.split(".")[0]
        if mod in ALLOWED_IMPORT_MODULES:
            continue
        assert root not in FORBIDDEN_IMPORT_ROOTS, f"forbidden import root: {mod}"


def test_A_no_forbidden_qualified_imports():
    assert not (_imported_modules() & FORBIDDEN_QUALIFIED)


def test_A_no_router_extractor_subagents_delegate_imports():
    mods = _imported_modules()
    assert "agent.subagent_router" not in mods
    assert "agent.subagent_request_extractor" not in mods
    assert "agent.subagents" not in mods
    assert "tools.delegate_tool" not in mods


def test_A_no_route_or_delegate_calls():
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            assert name not in {"route_named_subagent", "delegate_task"}


# =====================================================================
# Group B -- data model
# =====================================================================

def test_B_profileclear_frozen():
    pc = ProfileClear(dimension="env_mutation_requested", reason="r")
    with pytest.raises(dataclasses.FrozenInstanceError):
        pc.dimension = "execution_requested"


def test_B_confirmationprofile_frozen():
    prof = profiles._PROFILES["no_env_mutation"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        prof.name = "other"


def test_B_profile_clears_are_tuples():
    for prof in profiles._PROFILES.values():
        assert isinstance(prof.clears, tuple)


def test_B_registry_immutable_from_public_path():
    # No public mutable registry is exported.
    assert not hasattr(profiles, "PROFILES")
    assert isinstance(KNOWN_PROFILE_NAMES, frozenset)
    # The private registry itself is a read-only mapping proxy.
    with pytest.raises(TypeError):
        profiles._PROFILES["x"] = profiles._PROFILES["no_env_mutation"]


def test_B_no_timestamp_fields():
    for cls in (ProfileClear, ConfirmationProfile):
        names = {f.name for f in dataclasses.fields(cls)}
        assert not any("time" in n or "stamp" in n or "clock" in n for n in names)


def test_B_source_not_caller_supplied():
    profile_fields = {f.name for f in dataclasses.fields(ConfirmationProfile)}
    clear_fields = {f.name for f in dataclasses.fields(ProfileClear)}
    assert "source" not in profile_fields
    assert "source" not in clear_fields


def test_B_source_fixed_central_policy_in_output():
    posture = build_posture_by_name("no_env_mutation")
    assert all(a.source is ConfirmationSource.CENTRAL_POLICY for a in posture.assertions)


# =====================================================================
# Group C -- valid producer behavior
# =====================================================================

def test_C_build_by_name_returns_safeposture():
    assert isinstance(build_posture_by_name("no_env_mutation"), SafePosture)


def test_C_build_from_profile_returns_safeposture():
    prof = profiles._PROFILES["no_env_mutation"]
    assert isinstance(build_posture_from_profile(prof), SafePosture)


def test_C_every_element_is_confirmationassertion():
    posture = build_posture_by_name("no_env_mutation")
    assert all(isinstance(a, ConfirmationAssertion) for a in posture.assertions)


def test_C_every_assertion_source_central_policy():
    posture = build_posture_by_name("no_env_mutation")
    assert all(a.source is ConfirmationSource.CENTRAL_POLICY for a in posture.assertions)


def test_C_every_assertion_value_false():
    posture = build_posture_by_name("no_env_mutation")
    assert all(a.value is SignalValue.FALSE for a in posture.assertions)


def test_C_only_explicit_dimensions_emitted():
    posture = build_posture_by_name("no_env_mutation")
    assert [a.dimension for a in posture.assertions] == ["env_mutation_requested"]


def test_C_absent_dimensions_remain_unknown_after_merge():
    posture = build_posture_by_name("no_env_mutation")
    merged, _audit = merge_safety_posture(RequestSignals(), posture)
    assert merged.env_mutation_requested is SignalValue.FALSE
    # Every other safety dimension stayed UNKNOWN (not emitted, not cleared).
    for dim in T1 + T2 + T3:
        if dim == "env_mutation_requested":
            continue
        assert getattr(merged, dim) is SignalValue.UNKNOWN


def test_C_reason_copied_verbatim():
    expected = "central policy profile asserts no environment mutation request."
    posture = build_posture_by_name("no_env_mutation")
    assert posture.assertions[0].reason == expected


def test_C_deterministic_canonical_order_independent_of_authoring():
    a = ConfirmationProfile(
        name="p",
        clears=(
            ProfileClear("env_mutation_requested", "r1"),
            ProfileClear("execution_requested", "r2"),
        ),
    )
    b = ConfirmationProfile(
        name="p",
        clears=(
            ProfileClear("execution_requested", "r2"),
            ProfileClear("env_mutation_requested", "r1"),
        ),
    )
    dims_a = [x.dimension for x in build_posture_from_profile(a).assertions]
    dims_b = [x.dimension for x in build_posture_from_profile(b).assertions]
    assert dims_a == dims_b == ["execution_requested", "env_mutation_requested"]


# =====================================================================
# Group D -- rejection behavior (all loud)
# =====================================================================

def test_D_t3_dimension_rejected():
    prof = ConfirmationProfile(
        name="p", clears=(ProfileClear("secret_or_forbidden_path_referenced", "r"),)
    )
    with pytest.raises(ValueError):
        build_posture_from_profile(prof)


def test_D_unknown_dimension_rejected():
    prof = ConfirmationProfile(name="p", clears=(ProfileClear("not_a_field", "r"),))
    with pytest.raises(ValueError):
        build_posture_from_profile(prof)


def test_D_non_safety_intent_field_rejected():
    prof = ConfirmationProfile(name="p", clears=(ProfileClear("public_url_present", "r"),))
    with pytest.raises(ValueError):
        build_posture_from_profile(prof)


def test_D_blanket_all_safe_rejected():
    # Clearing every safety dimension (incl. T3) is impossible -- T3 is rejected.
    clears = tuple(ProfileClear(dim, "r") for dim in (T1 + T2 + T3))
    prof = ConfirmationProfile(name="p", clears=clears)
    with pytest.raises(ValueError):
        build_posture_from_profile(prof)
    # And there is no all-safe convenience constructor on the module.
    assert not any(
        n for n in dir(profiles) if "all_safe" in n.lower() or "clear_all" in n.lower()
    )


def test_D_no_source_field_available():
    profile_fields = {f.name for f in dataclasses.fields(ConfirmationProfile)}
    clear_fields = {f.name for f in dataclasses.fields(ProfileClear)}
    assert "source" not in profile_fields and "source" not in clear_fields


def test_D_empty_or_whitespace_reason_rejected():
    prof = ConfirmationProfile(
        name="p", clears=(ProfileClear("env_mutation_requested", "   "),)
    )
    with pytest.raises(ValueError):
        build_posture_from_profile(prof)


def test_D_duplicate_dimension_rejected():
    prof = ConfirmationProfile(
        name="p",
        clears=(
            ProfileClear("env_mutation_requested", "r1"),
            ProfileClear("env_mutation_requested", "r2"),
        ),
    )
    with pytest.raises(ValueError):
        build_posture_from_profile(prof)


def test_D_empty_clears_rejected():
    prof = ConfirmationProfile(name="p", clears=())
    with pytest.raises(ValueError):
        build_posture_from_profile(prof)


# =====================================================================
# Group E -- merge integration (test-only)
# =====================================================================

def test_E_producer_output_accepted_by_merge():
    posture = build_posture_by_name("no_env_mutation")
    merged, audit = merge_safety_posture(RequestSignals(), posture)
    assert isinstance(merged, RequestSignals)
    assert audit.entries


def test_E_nl_unknown_plus_profile_false_clears_to_false():
    posture = build_posture_by_name("no_env_mutation")
    merged, _audit = merge_safety_posture(RequestSignals(), posture)
    assert merged.env_mutation_requested is SignalValue.FALSE


def test_E_nl_true_plus_profile_false_remains_true():
    posture = build_posture_by_name("no_env_mutation")
    nl = RequestSignals(env_mutation_requested=SignalValue.TRUE)
    merged, _audit = merge_safety_posture(nl, posture)
    assert merged.env_mutation_requested is SignalValue.TRUE


def test_E_producer_never_emits_true_or_unknown():
    posture = build_posture_from_profile(_make_maximal_t1_t2_profile())
    assert all(a.value is SignalValue.FALSE for a in posture.assertions)


def _audit_entry(audit, dimension):
    for e in audit.entries:
        if e.dimension == dimension:
            return e
    raise AssertionError(f"no audit entry for {dimension}")


def test_E_audit_source_central_policy():
    posture = build_posture_by_name("no_env_mutation")
    _merged, audit = merge_safety_posture(RequestSignals(), posture)
    entry = _audit_entry(audit, "env_mutation_requested")
    assert entry.source is ConfirmationSource.CENTRAL_POLICY


def test_E_audit_effect_false_applied_when_nl_unknown():
    posture = build_posture_by_name("no_env_mutation")
    _merged, audit = merge_safety_posture(RequestSignals(), posture)
    entry = _audit_entry(audit, "env_mutation_requested")
    assert entry.effect is AuditEffect.POSTURE_FALSE_APPLIED


def test_E_audit_effect_overridden_when_nl_true():
    posture = build_posture_by_name("no_env_mutation")
    nl = RequestSignals(env_mutation_requested=SignalValue.TRUE)
    _merged, audit = merge_safety_posture(nl, posture)
    entry = _audit_entry(audit, "env_mutation_requested")
    assert entry.effect is AuditEffect.POSTURE_FALSE_OVERRIDDEN_BY_NL_TRUE
    assert entry.overridden_by_nl_true is True


def test_E_reason_remains_inert():
    posture = build_posture_by_name("no_env_mutation")
    merged, audit = merge_safety_posture(RequestSignals(), posture)
    entry = _audit_entry(audit, "env_mutation_requested")
    assert entry.reason == "central policy profile asserts no environment mutation request."
    # Reason did not influence the merged value (clear still applied normally).
    assert merged.env_mutation_requested is SignalValue.FALSE


# =====================================================================
# Group F -- router safety (test-only)
# =====================================================================

def test_F_maximal_profile_leaves_t3_unknown():
    posture = build_posture_from_profile(_make_maximal_t1_t2_profile())
    merged, _audit = merge_safety_posture(RequestSignals(), posture)
    assert merged.secret_or_forbidden_path_referenced is SignalValue.UNKNOWN
    for dim in T1 + T2:
        assert getattr(merged, dim) is SignalValue.FALSE


def test_F_descriptor_maps_t3_unknown_to_true():
    posture = build_posture_from_profile(_make_maximal_t1_t2_profile())
    merged, _audit = merge_safety_posture(RequestSignals(), posture)
    td = build_task_descriptor_from_signals(merged)
    assert td.targets_secret_or_forbidden_path is True


def test_F_router_returns_r0_none_after_maximal_profile():
    posture = build_posture_from_profile(_make_maximal_t1_t2_profile())
    merged, _audit = merge_safety_posture(RequestSignals(), posture)
    td = build_task_descriptor_from_signals(merged)
    decision = select_named_subagent_for_task(td)
    assert decision.selected_wrapper is None
    assert decision.fired_rule == "R0"


def test_F_no_wrapper_selected():
    posture = build_posture_from_profile(_make_maximal_t1_t2_profile())
    merged, _audit = merge_safety_posture(RequestSignals(), posture)
    td = build_task_descriptor_from_signals(merged)
    assert select_named_subagent_for_task(td).selected_wrapper is None


def test_F_no_route_or_delegate_symbols_in_production_source():
    # The production module never calls route_named_subagent / delegate_task
    # (docstring mentions are fine; this checks executable call sites only).
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            assert name not in {"route_named_subagent", "delegate_task"}


# =====================================================================
# Group G -- source-spoofing caveat documentation
# =====================================================================

def test_G_docstring_states_sanctioned_mint_path():
    doc = (profiles.__doc__ or "").lower()
    assert "sanctioned" in doc and "mint" in doc


def test_G_docstring_states_cannot_prove_authenticity():
    doc = (profiles.__doc__ or "").lower()
    assert "authenticity" in doc and "cannot" in doc


def test_G_confirmationassertion_constructor_remains_public():
    # Directly constructible out of band -- documents the honest residual.
    a = ConfirmationAssertion(
        dimension="env_mutation_requested",
        value=SignalValue.FALSE,
        source=ConfirmationSource.CENTRAL_POLICY,
        reason="hand-built",
    )
    assert a.source is ConfirmationSource.CENTRAL_POLICY


def test_G_no_token_gating_mechanism():
    # Producers take only their declared inputs -- no token / capability param.
    assert list(inspect.signature(build_posture_from_profile).parameters) == ["profile"]
    assert list(inspect.signature(build_posture_by_name).parameters) == ["profile_name"]
    assert not any("token" in n.lower() for n in dir(profiles))


# =====================================================================
# Group H -- registry / profile names
# =====================================================================

def test_H_known_profile_names_exact():
    assert KNOWN_PROFILE_NAMES == frozenset({"no_env_mutation"})


def test_H_unknown_profile_name_rejected():
    with pytest.raises(ValueError):
        build_posture_by_name("does_not_exist")


def test_H_non_str_and_empty_profile_name_rejected():
    with pytest.raises(ValueError):
        build_posture_by_name("")
    with pytest.raises(ValueError):
        build_posture_by_name("   ")
    with pytest.raises(ValueError):
        build_posture_by_name(None)  # type: ignore[arg-type]


def test_H_registry_values_are_profiles():
    assert all(isinstance(v, ConfirmationProfile) for v in profiles._PROFILES.values())


def test_H_output_independent_of_iteration_order():
    assert build_posture_by_name("no_env_mutation") == build_posture_by_name("no_env_mutation")


# =====================================================================
# Group I -- anti-regression
# =====================================================================

def test_I_t3_absent_from_every_production_profile():
    for prof in profiles._PROFILES.values():
        for clear in prof.clears:
            assert clear.dimension not in T3


def test_I_no_production_profile_clears_full_t1_t2():
    full = set(T1 + T2)
    for prof in profiles._PROFILES.values():
        cleared = {c.dimension for c in prof.clears}
        assert cleared != full


def test_I_gates_remain_inert_in_false_clear_map():
    assert cp._FALSE_CLEAR_BY_SOURCE[ConfirmationSource.PREFLIGHT_GATE] == frozenset()
    assert cp._FALSE_CLEAR_BY_SOURCE[ConfirmationSource.STATIC_SECURITY_GATE] == frozenset()


def test_I_false_clear_map_unchanged_by_profiles_import():
    assert cp._FALSE_CLEAR_BY_SOURCE[ConfirmationSource.CENTRAL_POLICY] == frozenset(T1 + T2)
    assert cp._FALSE_CLEAR_BY_SOURCE[ConfirmationSource.HUMAN_EXPLICIT] == frozenset(T1)


def test_I_validate_assertion_still_backstops_t3():
    with pytest.raises(ValueError):
        validate_assertion(
            "secret_or_forbidden_path_referenced",
            SignalValue.FALSE,
            ConfirmationSource.CENTRAL_POLICY,
            "r",
        )


def test_I_pure_nl_still_cannot_route():
    signals = extract_request_signals_from_text("search the web for python docs")
    td = build_task_descriptor_from_signals(signals)
    assert select_named_subagent_for_task(td).selected_wrapper is None


def test_I_safety_fields_order_membership_matches_policy():
    # Membership-only drift guard. The profile module's local emission-order
    # tuple must cover exactly the policy module's safety-field set. Order is
    # intentionally NOT asserted: _SAFETY_FIELDS_ORDER controls assertion
    # emission order only, while merge/routing resolve per dimension over
    # cp._SAFETY_FIELDS, so the local grouped (T1->T2->T3) order is acceptable.
    order = profiles._SAFETY_FIELDS_ORDER
    assert set(order) == set(cp._SAFETY_FIELDS)
    # No duplicate entries in the local emission-order tuple.
    assert len(order) == len(set(order))
    # Explicit missing / extra diagnostics for a clearer failure message.
    assert not (set(cp._SAFETY_FIELDS) - set(order)), "missing safety field(s)"
    assert not (set(order) - set(cp._SAFETY_FIELDS)), "extra non-safety field(s)"
