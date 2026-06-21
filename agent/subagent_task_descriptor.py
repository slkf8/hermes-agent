"""Deterministic normalization layer: RequestSignals -> TaskDescriptor (M11).

Pure, side-effect-free mapping. Given a *structured* signal object (explicit,
pre-classified tri-state signals -- never raw natural-language text as
authority), this module produces exactly one :class:`TaskDescriptor` for the
Central Router to evaluate.

Hard boundaries (see M11-1..M11-6 design artifacts):

* The normalizer returns exactly one ``TaskDescriptor``. It never returns a
  ``RouterDecision``, never selects a wrapper, and never emits wrapper names,
  toolsets, roles, command strings, delegate_task arguments, subtasks, or
  sequences.
* It performs no I/O: it does not read files, dereference refs, use the
  network, or start runtime.
* It owns no capability and no enforcement. It does **not** import or call
  ``route_named_subagent``, ``validate_raw_named_subagent_request``,
  ``delegate_task``, ``tools.delegate_tool``, or ``toolsets``.

Fail-closed asymmetry:

* Intent signals are fail-closed by omission:
  TRUE -> True; FALSE -> False; UNKNOWN -> False.
* Safety signals are fail-closed by inclusion:
  TRUE -> True; FALSE -> False; UNKNOWN -> True.

Because every safety field defaults to ``SignalValue.UNKNOWN``, an all-default
``RequestSignals`` maps every R0 safety flag to True and therefore routes to
R0 / no wrapper. Callers must explicitly mark safety fields ``FALSE`` to assert
a confidently-safe context; the normalizer never silently assumes safety
absence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Read-only coupling: import only the router's data classes (the descriptor
# shape). No enforcement, delegation, toolset, or runtime imports.
from agent.subagent_router import DeclaredRef, TaskDescriptor


class SignalValue(Enum):
    """Tri-state value for a single structured signal."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class SignalSource(Enum):
    """Provenance of a signal. Metadata only -- never affects mapping."""

    EXPLICIT_FLAG = "explicit_flag"
    DECLARED_REF = "declared_ref"
    DEFAULT = "default"


# Recognized declared-ref type vocabulary (awareness only; the normalizer does
# not enforce the enforcement layer's ref policy).
REF_PUBLIC_URL = "public_url"
REF_ALLOWLISTED_FILE = "allowlisted_file"
REF_PROVIDED_TEXT = "provided_text"
REF_REPORT_ARTIFACT = "report_artifact"
REF_UNKNOWN = "unknown"


@dataclass(frozen=True)
class RequestRef:
    """A pre-classified input reference. Never dereferenced or read."""

    type: str
    ref: str
    forbidden: bool = False


@dataclass(frozen=True)
class RequestSignals:
    """Structured, explicit signals describing a request.

    Intent signals default to ``UNKNOWN`` (mapped to False -> no route).
    Safety signals default to ``UNKNOWN`` (mapped to True -> R0 dominance).
    """

    # --- intent signals (fail-closed by omission) ---
    public_url_present: SignalValue = SignalValue.UNKNOWN
    open_web_search: SignalValue = SignalValue.UNKNOWN
    local_file_ref_present: SignalValue = SignalValue.UNKNOWN
    candidate_change_present: SignalValue = SignalValue.UNKNOWN
    judge_requested: SignalValue = SignalValue.UNKNOWN
    test_plan_requested: SignalValue = SignalValue.UNKNOWN
    symptom_present: SignalValue = SignalValue.UNKNOWN
    broad_inspection_requested: SignalValue = SignalValue.UNKNOWN
    gate_question_present: SignalValue = SignalValue.UNKNOWN
    gate_execute_requested: SignalValue = SignalValue.UNKNOWN

    # --- safety signals (fail-closed by inclusion) ---
    mutation_requested: SignalValue = SignalValue.UNKNOWN
    execution_requested: SignalValue = SignalValue.UNKNOWN
    runtime_start_requested: SignalValue = SignalValue.UNKNOWN
    real_delegate_requested: SignalValue = SignalValue.UNKNOWN
    secret_or_forbidden_path_referenced: SignalValue = SignalValue.UNKNOWN
    external_surface_requested: SignalValue = SignalValue.UNKNOWN
    external_connect_requested: SignalValue = SignalValue.UNKNOWN
    git_mutation_requested: SignalValue = SignalValue.UNKNOWN
    env_mutation_requested: SignalValue = SignalValue.UNKNOWN

    # --- auxiliary, non-authoritative ---
    declared_refs: list[RequestRef] = field(default_factory=list)
    declared_intent: Optional[str] = None


def _intent(value: SignalValue) -> bool:
    """Intent mapping: only an explicit TRUE yields True."""
    return value is SignalValue.TRUE


def _safety(value: SignalValue) -> bool:
    """Safety mapping: only an explicit FALSE yields False; else True."""
    return value is not SignalValue.FALSE


def build_task_descriptor_from_signals(signals: RequestSignals) -> TaskDescriptor:
    """Map structured ``RequestSignals`` to exactly one ``TaskDescriptor``.

    Pure and deterministic. Does not select a wrapper, evaluate any routing
    rule, read files, dereference refs, or call enforcement/delegation.
    """
    # Intent fields (fail-closed by omission).
    has_public_url = _intent(signals.public_url_present)
    is_open_web_search = _intent(signals.open_web_search)
    has_local_file_ref = _intent(signals.local_file_ref_present)
    has_candidate_change = _intent(signals.candidate_change_present)
    is_judge_ask = _intent(signals.judge_requested)
    deliverable_is_test_plan = _intent(signals.test_plan_requested)
    names_symptom = _intent(signals.symptom_present)
    is_broad_inspection = _intent(signals.broad_inspection_requested)
    is_gate_question = _intent(signals.gate_question_present)
    is_gate_execute = _intent(signals.gate_execute_requested)

    # Safety fields (fail-closed by inclusion).
    targets_secret_or_forbidden_path = _safety(signals.secret_or_forbidden_path_referenced)

    # Declared refs: copy-only conversion. Certain ref types/flags raise the
    # corresponding descriptor flags. Refs are never opened or resolved.
    declared_refs: list[DeclaredRef] = []
    for r in signals.declared_refs:
        declared_refs.append(DeclaredRef(type=r.type, ref=r.ref))
        if r.forbidden:
            targets_secret_or_forbidden_path = True
        if r.type == REF_PUBLIC_URL:
            has_public_url = True
        elif r.type == REF_ALLOWLISTED_FILE:
            has_local_file_ref = True
        # provided_text, report_artifact, unknown -> copied, raise no flag.

    return TaskDescriptor(
        has_public_url=has_public_url,
        is_open_web_search=is_open_web_search,
        has_local_file_ref=has_local_file_ref,
        has_candidate_change=has_candidate_change,
        is_judge_ask=is_judge_ask,
        deliverable_is_test_plan=deliverable_is_test_plan,
        names_symptom=names_symptom,
        is_broad_inspection=is_broad_inspection,
        is_gate_question=is_gate_question,
        is_gate_execute=is_gate_execute,
        is_mutating_request=_safety(signals.mutation_requested),
        is_exec_request=_safety(signals.execution_requested),
        is_runtime_start=_safety(signals.runtime_start_requested),
        is_real_delegate=_safety(signals.real_delegate_requested),
        targets_secret_or_forbidden_path=targets_secret_or_forbidden_path,
        is_external_surface=_safety(signals.external_surface_requested),
        is_external_connect=_safety(signals.external_connect_requested),
        is_git_mutation=_safety(signals.git_mutation_requested),
        is_env_mutation=_safety(signals.env_mutation_requested),
        declared_refs=declared_refs,
        # Copied verbatim; non-authoritative. Does not affect any field above.
        declared_intent=signals.declared_intent,
    )
