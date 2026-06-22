"""Sanctioned ``CENTRAL_POLICY`` confirmation-posture producer (M15).

This module is the single *sanctioned mint path* that deterministically builds a
:class:`~agent.subagent_confirmation_policy.SafePosture` from a static,
declarative :class:`ConfirmationProfile`. It is the only intended producer that
constructs ``CENTRAL_POLICY`` safety clearances for the named-subagent pipeline.

Hard boundaries:

* Pure and deterministic. No I/O, no filesystem, no env, no network, no
  subprocess, no clock, no randomness.
* Clears T1/T2 safety dimensions only. It never clears T3
  (``secret_or_forbidden_path_referenced``); a profile that names a T3 field is
  rejected loudly through the :func:`validate_assertion` backstop.
* No runtime, no ``route_named_subagent``, no ``delegate_task``. It is never a
  live caller of ``merge_safety_posture`` -- it only *returns* an inert
  ``SafePosture`` value, which a separate authorized stage may later consume.
* It does not import the router, the extractor, the enforcement layer
  (``agent.subagents``), ``tools.delegate_tool``, or any runtime/gateway/CLI/
  messaging/transport surface.

Source-spoofing residual (stated honestly):

* Every assertion this producer emits is minted with
  ``ConfirmationSource.CENTRAL_POLICY`` through the public
  ``ConfirmationAssertion`` constructor (which **remains public** in M15) so
  that :func:`validate_assertion` always runs. Centralizing minting here is the
  closest pure-Python approximation of provenance, but it is not proof: pure
  Python cannot prove the authenticity of ``source``. A caller can still
  hand-construct a ``CENTRAL_POLICY`` assertion out of band, so authenticity
  remains a *contractual* guarantee, not a constructive one. No private or
  token-gated constructor is introduced in M15.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from agent.subagent_confirmation_policy import (
    ConfirmationAssertion,
    ConfirmationSource,
    SafePosture,
    validate_assertion,
)
from agent.subagent_task_descriptor import SignalValue

__all__ = [
    "ProfileClear",
    "ConfirmationProfile",
    "validate_profile",
    "build_posture_from_profile",
    "build_posture_by_name",
    "KNOWN_PROFILE_NAMES",
]


# Canonical emission order for assertions, grouped T1 -> T2 -> T3. T3 is never
# emitted by a valid profile; it appears here only so the ordering key is total
# over every safety dimension. Output order is independent of profile authoring
# order, which keeps the producer deterministic.
_SAFETY_FIELDS_ORDER: tuple[str, ...] = (
    "mutation_requested",
    "execution_requested",
    "runtime_start_requested",
    "git_mutation_requested",
    "env_mutation_requested",
    "real_delegate_requested",
    "external_surface_requested",
    "external_connect_requested",
    "secret_or_forbidden_path_referenced",
)

_ORDER_INDEX: dict[str, int] = {name: i for i, name in enumerate(_SAFETY_FIELDS_ORDER)}


@dataclass(frozen=True)
class ProfileClear:
    """One declarative per-dimension clearance.

    ``dimension`` is a safety field name; ``reason`` is a static, audit-only
    string copied verbatim into the minted assertion. There is no ``source``
    field -- the source is fixed downstream to ``CENTRAL_POLICY``.
    """

    dimension: str
    reason: str


@dataclass(frozen=True)
class ConfirmationProfile:
    """A named, immutable, declarative set of per-dimension clearances.

    There is no blanket "all-safe" constructor and no ``source``/timestamp
    field; each cleared field must be listed explicitly as a ``ProfileClear``.
    """

    name: str
    clears: tuple[ProfileClear, ...]


def validate_profile(profile: ConfirmationProfile) -> None:
    """Validate a confirmation profile, raising loudly on any violation.

    Profile-level checks (type, non-empty name, non-empty clears, duplicate
    dimensions) are enforced here; dimension validity, the T1/T2-only tier rule
    (which rejects T3 and unknown/non-safety dimensions), and the non-empty
    reason rule are delegated to :func:`validate_assertion` so the policy module
    stays the single source of truth for the clear-permission matrix.
    """
    if not isinstance(profile, ConfirmationProfile):
        raise TypeError(
            f"profile must be a ConfirmationProfile, got {type(profile).__name__}"
        )
    if not isinstance(profile.name, str) or not profile.name.strip():
        raise ValueError("profile name must be a non-empty string")
    if not isinstance(profile.clears, tuple):
        raise TypeError("profile.clears must be a tuple")
    if not profile.clears:
        raise ValueError("profile.clears must be non-empty")

    seen: set[str] = set()
    for clear in profile.clears:
        if not isinstance(clear, ProfileClear):
            raise TypeError("each clear must be a ProfileClear")
        if not isinstance(clear.dimension, str):
            raise TypeError("clear.dimension must be a string")
        if not isinstance(clear.reason, str) or not clear.reason.strip():
            raise ValueError("clear.reason must be a non-empty string")
        if clear.dimension in seen:
            raise ValueError(f"duplicate dimension in profile: {clear.dimension!r}")
        seen.add(clear.dimension)
        # Authoritative backstop: the dimension must be a known safety field
        # that CENTRAL_POLICY may clear. This rejects T3 and any unknown or
        # non-safety dimension loudly.
        validate_assertion(
            clear.dimension,
            SignalValue.FALSE,
            ConfirmationSource.CENTRAL_POLICY,
            clear.reason,
        )


def build_posture_from_profile(profile: ConfirmationProfile) -> SafePosture:
    """Build a deterministic ``CENTRAL_POLICY`` ``SafePosture`` from a profile.

    Validates the profile first, then emits exactly one ``ConfirmationAssertion``
    per ``ProfileClear`` -- each with ``value=SignalValue.FALSE`` and
    ``source=ConfirmationSource.CENTRAL_POLICY``, constructed through the normal
    (public) constructor so ``validate_assertion`` runs. Assertions are ordered
    canonically by ``_SAFETY_FIELDS_ORDER``; absent dimensions are not emitted.
    Pure: no I/O, no runtime, no route/delegate, no live merge call.
    """
    validate_profile(profile)
    ordered = sorted(profile.clears, key=lambda c: _ORDER_INDEX[c.dimension])
    assertions = tuple(
        ConfirmationAssertion(
            dimension=c.dimension,
            value=SignalValue.FALSE,
            source=ConfirmationSource.CENTRAL_POLICY,
            reason=c.reason,
        )
        for c in ordered
    )
    return SafePosture(assertions=assertions)


# Static, in-code profile registry. Exactly one conservative production profile:
# a single T1 clearance asserting no environment mutation request. No maximal
# T1/T2 profile is shipped here (maximal profiles exist only in tests).
_PROFILES = MappingProxyType(
    {
        "no_env_mutation": ConfirmationProfile(
            name="no_env_mutation",
            clears=(
                ProfileClear(
                    dimension="env_mutation_requested",
                    reason=(
                        "central policy profile asserts no environment mutation "
                        "request."
                    ),
                ),
            ),
        ),
    }
)

# Fail-closed at import: every registered profile must be valid.
for _profile in _PROFILES.values():
    validate_profile(_profile)
del _profile

KNOWN_PROFILE_NAMES = frozenset(_PROFILES)


def build_posture_by_name(profile_name: str) -> SafePosture:
    """Build a ``SafePosture`` for a registered profile name.

    Rejects a non-string, empty/whitespace, or unknown ``profile_name`` loudly.
    """
    if not isinstance(profile_name, str) or not profile_name.strip():
        raise ValueError("profile_name must be a non-empty string")
    try:
        profile = _PROFILES[profile_name]
    except KeyError:
        raise ValueError(f"unknown profile name: {profile_name!r}") from None
    return build_posture_from_profile(profile)
