"""Explicit, auditable, non-NL safety-confirmation policy (M13).

This module is the *only* sanctioned producer of safety ``SignalValue.FALSE``
values for the named-subagent pipeline. The deterministic NL extractor
(:mod:`agent.subagent_request_extractor`) emits only ``TRUE``/``UNKNOWN`` and can
therefore never clear a safety field; every safety field defaults to ``UNKNOWN``
which the normalizer maps to ``True`` -> R0 dominance. A confidently-safe request
can only become eligible for routing if an explicit, auditable, *non-NL* source
asserts ``FALSE`` on the relevant safety dimensions. This module models those
assertions and merges them into a ``RequestSignals``.

Hard boundaries (see M13-1..M13-5 design artifacts):

* Pure and deterministic. No I/O, no clock, no randomness, no environment reads.
* It never routes, never selects a wrapper, never emits a ``TaskDescriptor`` or
  ``RouterDecision``, and never calls ``route_named_subagent`` or
  ``delegate_task``. It does not import the router, the extractor, the
  enforcement layer, toolsets, or any runtime/gateway/CLI/messaging surface.
* It is *not* sourced from raw natural language or ``declared_intent``. The only
  free text it carries (``reason``) is audit-only and is never parsed.

Merge asymmetry (danger-dominant):

* NL ``TRUE`` dominates a confirmation ``FALSE`` -- detected danger is never
  cleared.
* A confirmation ``FALSE`` may only clear an ``UNKNOWN`` (NL never emits
  ``FALSE``, so the only ``FALSE`` source is a posture).
* A confirmation ``TRUE`` may raise danger and is always honored.
* Intent fields, ``declared_refs`` and ``declared_intent`` pass through
  unchanged.

Tiering / MVP scope:

* T1 (capability-absence) is clearable by ``CENTRAL_POLICY`` and, in the data
  model, by ``HUMAN_EXPLICIT`` (no collection channel exists yet).
* T2 (structural leaf-wrapper invariants) is clearable by ``CENTRAL_POLICY``
  only.
* T3 (``secret_or_forbidden_path_referenced``) is **non-clearable by any source
  in MVP**. Consequently this module alone can never unlock routing: a maximally
  clearing posture still leaves T3 ``UNKNOWN`` -> R0 / no wrapper.
* ``PREFLIGHT_GATE`` / ``STATIC_SECURITY_GATE`` are represented in the enum but
  clear nothing in MVP.

Audit and provenance:

* The ``AuditTrail`` is **output-only**: it records, per safety dimension, the
  NL value, the posture contribution, the final merged value, the contributing
  source/reason, and a summary effect. It is never consumed by the normalizer or
  router and can never influence routing.
* ``reason`` is **inert**: it is copied verbatim into the audit and is never
  parsed, matched, or used as a signal -- this prevents an NL backdoor.
* **Source-spoofing caveat:** in pure Python the ``source`` on an assertion is
  caller-settable; this module cannot cryptographically prove provenance. Trust
  that a privileged source (e.g. ``CENTRAL_POLICY``) was genuinely minted by the
  central-policy producer is a contractual guarantee to be enforced by a future
  sanctioned producer, not by construction here.
* No timestamp/clock is recorded, preserving determinism.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

from agent.subagent_task_descriptor import RequestSignals, SignalValue

__all__ = [
    "ConfirmationSource",
    "ConfirmationAssertion",
    "SafePosture",
    "AuditEffect",
    "AuditEntry",
    "AuditTrail",
    "validate_assertion",
    "resolve_dimension",
    "merge_safety_posture",
]


# --- safety-field vocabulary (mirrors RequestSignals; kept local) ------------

# The 9 safety fields, in RequestSignals declaration order. A test asserts this
# tuple matches the safety subset of RequestSignals exactly.
_SAFETY_FIELDS: tuple[str, ...] = (
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

# T1 -- capability-absence; clearable for read-only/non-mutating profiles.
_T1_FIELDS: frozenset[str] = frozenset({
    "mutation_requested",
    "execution_requested",
    "runtime_start_requested",
    "git_mutation_requested",
    "env_mutation_requested",
})

# T2 -- structural leaf-wrapper invariants; central-policy structural clear only.
_T2_FIELDS: frozenset[str] = frozenset({
    "real_delegate_requested",
    "external_surface_requested",
    "external_connect_requested",
})

# T3 -- highest sensitivity; non-clearable by any source in MVP.
_T3_FIELDS: frozenset[str] = frozenset({
    "secret_or_forbidden_path_referenced",
})


class ConfirmationSource(Enum):
    """Provenance of a confirmation assertion. Never NL / MODEL / RUNTIME."""

    CENTRAL_POLICY = "central_policy"
    HUMAN_EXPLICIT = "human_explicit"
    PREFLIGHT_GATE = "preflight_gate"
    STATIC_SECURITY_GATE = "static_security_gate"


# Per-source set of fields that the source may *clear* (assert FALSE on). T3 is
# in no set, so it is non-clearable by construction. Gate sources clear nothing
# in MVP (modeled but inert). TRUE (raising danger) is never gated by this map.
_FALSE_CLEAR_BY_SOURCE: dict[ConfirmationSource, frozenset[str]] = {
    ConfirmationSource.CENTRAL_POLICY: _T1_FIELDS | _T2_FIELDS,
    ConfirmationSource.HUMAN_EXPLICIT: _T1_FIELDS,
    ConfirmationSource.PREFLIGHT_GATE: frozenset(),
    ConfirmationSource.STATIC_SECURITY_GATE: frozenset(),
}


def _false_clear_allowed(source: ConfirmationSource, dimension: str) -> bool:
    """Whether ``source`` may assert FALSE (clear) on ``dimension``."""
    return dimension in _FALSE_CLEAR_BY_SOURCE[source]


def validate_assertion(
    dimension: str,
    value: "SignalValue",
    source: "ConfirmationSource",
    reason: str,
) -> None:
    """Validate a confirmation assertion, raising ``ValueError`` loudly.

    Invalid input is never silently ignored. An assertion is rejected when its
    dimension is not a safety field, its value/source are of the wrong type, its
    reason is empty/whitespace, or it attempts a FALSE clear that its source's
    tier does not permit (notably any FALSE on a T3 field).
    """
    if dimension not in _SAFETY_FIELDS:
        raise ValueError(f"unknown safety dimension: {dimension!r}")
    if not isinstance(value, SignalValue):
        raise ValueError(f"value must be a SignalValue, got {type(value).__name__}")
    if not isinstance(source, ConfirmationSource):
        raise ValueError(
            f"source must be a ConfirmationSource, got {type(source).__name__}"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    if value is SignalValue.FALSE and not _false_clear_allowed(source, dimension):
        raise ValueError(
            f"source {source.name} may not clear (assert FALSE on) dimension "
            f"{dimension!r}"
        )


@dataclass(frozen=True)
class ConfirmationAssertion:
    """One explicit, auditable, per-field safety assertion.

    ``value`` FALSE clears, TRUE raises, UNKNOWN is inert. ``reason`` is required
    and is audit-only -- it is never parsed or used as a signal.
    """

    dimension: str
    value: "SignalValue"
    source: "ConfirmationSource"
    reason: str

    def __post_init__(self) -> None:
        validate_assertion(self.dimension, self.value, self.source, self.reason)


@dataclass(frozen=True)
class SafePosture:
    """An immutable collection of confirmation assertions.

    There is no blanket "all-safe" constructor; each cleared field must trace to
    at least one assertion. Resolution is strictly per-field with no cross-field
    inference.
    """

    assertions: tuple[ConfirmationAssertion, ...] = ()


class AuditEffect(Enum):
    """Summary label for how one safety dimension's value was decided.

    Audit-only. The effect explains the per-field merge outcome; it is recorded
    for forensic review and never influences the merged ``RequestSignals``.
    """

    POSTURE_FALSE_OVERRIDDEN_BY_NL_TRUE = "posture_false_overridden_by_nl_true"
    NL_TRUE_PRESERVED = "nl_true_preserved"
    POSTURE_TRUE_RAISED = "posture_true_raised"
    POSTURE_FALSE_APPLIED = "posture_false_applied"
    POSTURE_UNKNOWN_INERT = "posture_unknown_inert"
    NO_POSTURE_CONTRIBUTION = "no_posture_contribution"


@dataclass(frozen=True)
class AuditEntry:
    """The complete, self-contained audit record for one safety dimension.

    Records the NL input (``nl_value``), the posture's resolved contribution
    (``posture_value``), and the final ``merged_value`` that appears in the
    returned ``RequestSignals``, plus the contributing ``source``/``reason``, a
    summary ``effect``, and whether a posture clear was suppressed by an NL TRUE
    (``overridden_by_nl_true``). ``reason`` is copied verbatim and never
    interpreted. This record is output-only and never affects routing.
    """

    dimension: str
    nl_value: "SignalValue"
    posture_value: "SignalValue"
    merged_value: "SignalValue"
    source: Optional["ConfirmationSource"]
    reason: Optional[str]
    effect: "AuditEffect"
    overridden_by_nl_true: bool


@dataclass(frozen=True)
class AuditTrail:
    """One ``AuditEntry`` per safety dimension, in fixed ``_SAFETY_FIELDS`` order."""

    entries: tuple[AuditEntry, ...]


def _resolve_dimension_full(
    posture: SafePosture, dimension: str
) -> tuple["SignalValue", Optional["ConfirmationSource"], Optional[str], bool]:
    """Danger-dominant per-field resolution over a posture.

    TRUE wins over FALSE; UNKNOWN assertions are inert. Selection among equal
    winners is canonicalized by ``(source.value, reason)`` so the result is
    independent of assertion order. The fourth element, ``had_assertions``,
    reports whether any assertion targeted this dimension (used only to
    distinguish an inert UNKNOWN posture from no contribution at all).
    """
    relevant = [a for a in posture.assertions if a.dimension == dimension]
    had_assertions = bool(relevant)
    trues = sorted(
        (a for a in relevant if a.value is SignalValue.TRUE),
        key=lambda a: (a.source.value, a.reason),
    )
    if trues:
        a = trues[0]
        return SignalValue.TRUE, a.source, a.reason, had_assertions
    falses = sorted(
        (a for a in relevant if a.value is SignalValue.FALSE),
        key=lambda a: (a.source.value, a.reason),
    )
    if falses:
        a = falses[0]
        return SignalValue.FALSE, a.source, a.reason, had_assertions
    return SignalValue.UNKNOWN, None, None, had_assertions


def resolve_dimension(posture: SafePosture, dimension: str) -> "SignalValue":
    """Return the posture's resolved ``SignalValue`` for one safety dimension."""
    if dimension not in _SAFETY_FIELDS:
        raise ValueError(f"unknown safety dimension: {dimension!r}")
    resolved, _source, _reason, _had = _resolve_dimension_full(posture, dimension)
    return resolved


def _audit_effect(
    nl_value: "SignalValue", posture_value: "SignalValue", had_assertions: bool
) -> "AuditEffect":
    """Classify the per-field merge outcome for the audit trail.

    Deterministic and total; evaluated in fixed precedence. Audit-only -- this
    label never affects the merged ``RequestSignals``.
    """
    if nl_value is SignalValue.TRUE and posture_value is SignalValue.FALSE:
        return AuditEffect.POSTURE_FALSE_OVERRIDDEN_BY_NL_TRUE
    if nl_value is SignalValue.TRUE:
        return AuditEffect.NL_TRUE_PRESERVED
    if posture_value is SignalValue.TRUE:
        return AuditEffect.POSTURE_TRUE_RAISED
    if posture_value is SignalValue.FALSE:
        return AuditEffect.POSTURE_FALSE_APPLIED
    if had_assertions:
        return AuditEffect.POSTURE_UNKNOWN_INERT
    return AuditEffect.NO_POSTURE_CONTRIBUTION


def merge_safety_posture(
    nl: RequestSignals, posture: SafePosture
) -> tuple[RequestSignals, AuditTrail]:
    """Merge confirmation safety posture into NL-derived ``RequestSignals``.

    Pure and deterministic. Per safety field, danger-dominant combine:
    NL ``TRUE`` dominates; else a posture ``TRUE`` raises; else a posture
    ``FALSE`` clears an ``UNKNOWN``; else ``UNKNOWN``. Intent fields,
    ``declared_refs`` and ``declared_intent`` pass through unchanged. Returns a
    new ``RequestSignals`` plus an ``AuditTrail`` (never a ``TaskDescriptor``,
    ``RouterDecision``, or wrapper name).
    """
    merged_safety: dict[str, "SignalValue"] = {}
    entries: list[AuditEntry] = []
    for dimension in _SAFETY_FIELDS:
        resolved, source, reason, had_assertions = _resolve_dimension_full(
            posture, dimension
        )
        nl_value = getattr(nl, dimension)
        if nl_value is SignalValue.TRUE or resolved is SignalValue.TRUE:
            combined = SignalValue.TRUE
        elif resolved is SignalValue.FALSE:
            combined = SignalValue.FALSE
        else:
            combined = SignalValue.UNKNOWN
        merged_safety[dimension] = combined
        # Audit assembly is a read-only side-record: it reflects the values
        # computed above and never feeds back into ``merged_safety``.
        entries.append(
            AuditEntry(
                dimension=dimension,
                nl_value=nl_value,
                posture_value=resolved,
                merged_value=combined,
                source=source,
                reason=reason,
                effect=_audit_effect(nl_value, resolved, had_assertions),
                overridden_by_nl_true=(
                    resolved is SignalValue.FALSE and nl_value is SignalValue.TRUE
                ),
            )
        )
    merged = replace(nl, **merged_safety)
    return merged, AuditTrail(entries=tuple(entries))
