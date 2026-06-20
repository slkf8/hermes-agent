"""Deterministic Central Router policy selector (M10).

Internal policy only. Given a *structured* description of a task, this module
selects **zero or one** restricted named sub-agent wrapper and emits a
rationale trace explaining the decision. It is the policy layer that sits
*upstream* of the existing enforcement layer in ``agent.subagents``.

Hard boundaries (see M10-1..M10-6 design artifacts):

* Selection is a pure function of structured descriptors. There is **no**
  natural-language keyword matching, no model call, no clock, no randomness,
  and no I/O of any kind.
* The router returns a wrapper *name* (or ``None``) plus a rationale trace.
  It never returns toolsets, roles, command strings, delegate_task arguments,
  or sub-task lists.
* The router never executes a wrapper. It does **not** call
  ``route_named_subagent``, ``validate_raw_named_subagent_request``, or
  ``delegate_task``, and it does **not** import ``tools.delegate_tool``.
* Capability remains owned entirely by the enforcement layer. The only
  coupling to ``agent.subagents`` is a read of the ``_PRESETS`` *names* for
  compatibility validation.

Routing rules (first confident match wins; >=2 intent rules => no wrapper):

* R0  SAFETY PRE-FILTER        -> no wrapper
* R1  EXTERNAL KNOWLEDGE       -> @research
* R1b PUBLIC WEB + LOCAL FILE  -> no wrapper (Central decomposition)
* R2  GATE GO/NO-GO DECISION   -> @security_gate
* R3  PROPOSED CHANGE REVIEW   -> @patch_reviewer
* R4  TEST STRATEGY            -> @test_planner
* R5  DEFECT DIAGNOSIS         -> @debug_static
* R6  GENERAL STATIC AUDIT     -> @audit_static
* R7  DEFAULT / ambiguous / multiple -> no wrapper
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Read-only coupling: import the enforcement-layer preset *names* purely for
# compatibility validation. ``agent.subagents`` does not import
# ``tools.delegate_tool`` at module load (that import is lazy, inside
# ``_call_delegate_task``), so this import performs no delegation wiring.
from agent.subagents import _PRESETS


# --- Wrapper name domain (canonical) ---------------------------------------

RESEARCH = "@research"
AUDIT_STATIC = "@audit_static"
DEBUG_STATIC = "@debug_static"
TEST_PLANNER = "@test_planner"
PATCH_REVIEWER = "@patch_reviewer"
SECURITY_GATE = "@security_gate"

WRAPPER_NAMES: frozenset[str] = frozenset({
    RESEARCH,
    AUDIT_STATIC,
    DEBUG_STATIC,
    TEST_PLANNER,
    PATCH_REVIEWER,
    SECURITY_GATE,
})


def enforcement_preset_names() -> frozenset[str]:
    """Return the wrapper names known to the enforcement layer (``_PRESETS``).

    Exposed so tests can assert that the router's wrapper domain stays
    compatible with ``agent.subagents._PRESETS`` without the router ever
    calling an enforcement function.
    """
    return frozenset(_PRESETS)


# --- Fired-rule domain ------------------------------------------------------

RULE_IDS: frozenset[str] = frozenset({
    "R0", "R1", "R1b", "R2", "R3", "R4", "R5", "R6", "R7",
})


# --- Data contract ----------------------------------------------------------

@dataclass(frozen=True)
class DeclaredRef:
    """A declared input reference descriptor (shape only; not dereferenced)."""

    type: str
    ref: str


@dataclass(frozen=True)
class TaskDescriptor:
    """Structured, pre-normalized signals describing a task.

    Every field is a deterministic descriptor. The router reasons over these
    booleans/enums *only* -- it never inspects raw natural-language text.
    Normalizing free text into this descriptor is a separate, caller-owned
    concern, which keeps the router wording-agnostic.
    """

    # R1 / R1b -- external knowledge signals
    has_public_url: bool = False
    is_open_web_search: bool = False
    has_local_file_ref: bool = False

    # R3 -- proposed change review
    has_candidate_change: bool = False
    is_judge_ask: bool = False

    # R4 -- test strategy
    deliverable_is_test_plan: bool = False

    # R5 -- defect diagnosis
    names_symptom: bool = False

    # R6 -- general static audit
    is_broad_inspection: bool = False

    # R2 -- gate go/no-go
    is_gate_question: bool = False
    is_gate_execute: bool = False

    # R0 -- safety pre-filter signals
    is_mutating_request: bool = False
    is_exec_request: bool = False
    is_runtime_start: bool = False
    is_real_delegate: bool = False
    targets_secret_or_forbidden_path: bool = False
    is_external_surface: bool = False
    is_external_connect: bool = False
    is_git_mutation: bool = False
    is_env_mutation: bool = False

    # Auxiliary, non-authoritative context
    declared_refs: list[DeclaredRef] = field(default_factory=list)
    declared_intent: Optional[str] = None


@dataclass(frozen=True)
class RouterDecision:
    """The router output object.

    By construction this carries only a wrapper *name* (or ``None``) plus a
    rationale trace. It intentionally has no ``toolsets``, ``role``, command
    string, ``delegate_task`` argument, or sub-task fields.
    """

    selected_wrapper: Optional[str]
    fired_rule: Optional[str]
    positive_signals: list[str]
    disqualifiers_checked: list[str]
    multi_intent_detected: bool
    fail_closed_reason: Optional[str]
    reason: str


# --- R0 safety pre-filter ---------------------------------------------------

# Ordered so the reported class is stable/deterministic for a given descriptor.
_R0_CLASSES: tuple[tuple[str, str], ...] = (
    ("is_mutating_request", "write/edit/apply/modify"),
    ("is_exec_request", "execute/run command"),
    ("is_runtime_start", "runtime startup"),
    ("is_real_delegate", "real delegate_task"),
    ("targets_secret_or_forbidden_path", "secret/forbidden-path"),
    ("is_external_surface", "gateway/TUI/RPC/external-route exposure"),
    ("is_external_connect", "external service connection"),
    ("is_git_mutation", "git mutation"),
    ("is_env_mutation", "dependency/venv mutation"),
)


def _r0_safety_class(td: TaskDescriptor) -> Optional[str]:
    for attr, label in _R0_CLASSES:
        if getattr(td, attr):
            return label
    return None


# --- Decision assembly helpers ---------------------------------------------

def _select(
    wrapper: str,
    rule: str,
    positive_signals: list[str],
    disqualifiers_checked: list[str],
    reason: str,
) -> RouterDecision:
    # Defensive, deterministic compatibility guard. Cannot trigger for the
    # canonical constants, but if the wrapper domain ever drifts out of sync
    # with the enforcement layer we fail closed rather than emit an
    # unroutable name.
    if wrapper not in WRAPPER_NAMES or wrapper not in enforcement_preset_names():
        return _no_wrapper(
            fired_rule="R7",
            fail_closed_reason=f"R7: incompatible wrapper name {wrapper!r}",
            multi_intent=False,
            positive_signals=[],
            disqualifiers_checked=["wrapper_in_presets"],
            reason="selected wrapper is not compatible with _PRESETS",
        )
    return RouterDecision(
        selected_wrapper=wrapper,
        fired_rule=rule,
        positive_signals=list(positive_signals),
        disqualifiers_checked=list(disqualifiers_checked),
        multi_intent_detected=False,
        fail_closed_reason=None,
        reason=reason,
    )


def _no_wrapper(
    *,
    fired_rule: Optional[str],
    fail_closed_reason: str,
    multi_intent: bool,
    positive_signals: list[str],
    disqualifiers_checked: list[str],
    reason: str,
) -> RouterDecision:
    # Invariant: no wrapper => fail_closed_reason must be a non-empty string.
    if not fail_closed_reason:
        fail_closed_reason = "fail-closed: no wrapper selected"
    return RouterDecision(
        selected_wrapper=None,
        fired_rule=fired_rule,
        positive_signals=list(positive_signals),
        disqualifiers_checked=list(disqualifiers_checked),
        multi_intent_detected=multi_intent,
        fail_closed_reason=fail_closed_reason,
        reason=reason,
    )


# --- Public entry point -----------------------------------------------------

def select_named_subagent_for_task(
    task_descriptor: TaskDescriptor,
    *,
    request_metadata: Optional[dict] = None,
) -> RouterDecision:
    """Select zero or one named wrapper for a task, deterministically.

    ``request_metadata`` is accepted for interface stability but is treated as
    non-authoritative context; it never grants capability and does not affect
    selection in this milestone.
    """
    td = task_descriptor

    # R0 -- safety pre-filter pre-empts everything else.
    r0_class = _r0_safety_class(td)
    if r0_class is not None:
        return _no_wrapper(
            fired_rule="R0",
            fail_closed_reason=f"R0: {r0_class} requested",
            multi_intent=False,
            positive_signals=[r0_class],
            disqualifiers_checked=[],
            reason=f"R0 safety pre-filter: {r0_class} cannot be satisfied by any leaf wrapper",
        )

    web_signal = td.has_public_url or td.is_open_web_search

    # R1b -- public web combined with a local file ref is not a single-wrapper
    # task; Central must decompose it.
    if web_signal and td.has_local_file_ref:
        return _no_wrapper(
            fired_rule="R1b",
            fail_closed_reason="R1b: public web + local file refs -> Central decomposition",
            multi_intent=True,
            positive_signals=["web_signal", "has_local_file_ref"],
            disqualifiers_checked=[],
            reason="task mixes public-web and local-file inputs; not a single-wrapper task",
        )

    # R1..R6 -- collect every intent rule that fires. Mutual exclusivity is
    # enforced by counting, not by guards, so colliding intents are detected
    # as multi-intent rather than silently resolved.
    hits: list[tuple[str, str, list[str]]] = []

    if web_signal and not td.has_local_file_ref:
        hits.append(("R1", RESEARCH, ["web_signal"]))
    if td.is_gate_question and not td.is_gate_execute:
        hits.append(("R2", SECURITY_GATE, ["is_gate_question"]))
    if td.has_candidate_change and td.is_judge_ask:
        hits.append(("R3", PATCH_REVIEWER, ["has_candidate_change", "is_judge_ask"]))
    if td.deliverable_is_test_plan:
        hits.append(("R4", TEST_PLANNER, ["deliverable_is_test_plan"]))
    if td.names_symptom:
        hits.append(("R5", DEBUG_STATIC, ["names_symptom"]))
    if td.is_broad_inspection:
        hits.append(("R6", AUDIT_STATIC, ["is_broad_inspection"]))

    if len(hits) == 1:
        rule, wrapper, positives = hits[0]
        return _select(
            wrapper=wrapper,
            rule=rule,
            positive_signals=positives,
            disqualifiers_checked=_disqualifiers_for(rule),
            reason=f"{rule} fired uniquely -> {wrapper}",
        )

    if len(hits) >= 2:
        rules = "+".join(h[0] for h in hits)
        return _no_wrapper(
            fired_rule="R7",
            fail_closed_reason=f"R7: multi-intent ({rules})",
            multi_intent=True,
            positive_signals=[h[0] for h in hits],
            disqualifiers_checked=[],
            reason=f"multiple intent rules fired ({rules}); rejected as single-wrapper task",
        )

    # No intent rule fired.
    return _no_wrapper(
        fired_rule="R7",
        fail_closed_reason="R7: ambiguous, no rule fired",
        multi_intent=False,
        positive_signals=[],
        disqualifiers_checked=[],
        reason="no confident rule fired; Central handles directly or asks for scope",
    )


# Descriptors the router verifies absent/false to confirm a rule's exclusivity.
_DISQUALIFIERS = {
    "R1": ["has_local_file_ref"],
    "R2": ["is_gate_execute"],
    "R3": ["deliverable_is_test_plan", "names_symptom"],
    "R4": ["has_candidate_change", "names_symptom"],
    "R5": ["has_candidate_change", "is_broad_inspection"],
    "R6": ["names_symptom", "has_candidate_change"],
}


def _disqualifiers_for(rule: str) -> list[str]:
    return list(_DISQUALIFIERS.get(rule, []))
