"""Deterministic NL -> RequestSignals extractor (M12).

Pure, deterministic, *untrusted* signal proposer. Given a raw text string, it
produces one :class:`RequestSignals` by matching high-precision cues. It is the
left-most stage of the policy pipeline:

    text -> RequestSignals -> TaskDescriptor -> RouterDecision

Trust model and hard boundaries (see M12-1..M12-6 design artifacts):

* The extractor is an untrusted *proposer*, never an authorizer. It emits a
  ``RequestSignals`` only -- never a RouterDecision, wrapper name, toolset,
  role, execution argument, or delegate call.
* It performs no I/O: no file access (no ``open``), no filesystem (no ``os`` /
  ``pathlib``), no network, no subprocess. Refs are classified by string shape
  only and are never dereferenced, read, resolved, stat-ed, or validated.
* It imports only the input-side data classes. It does not import the router,
  enforcement, runtime, gateway, CLI, messaging, transports, toolsets, or
  delegate modules, and never calls ``route_named_subagent`` or
  ``delegate_task``.

Critical invariant -- raw NL may raise safety but may never clear it:

* Intent fields are only ``SignalValue.TRUE`` (explicit cue) or
  ``SignalValue.UNKNOWN`` (no cue). Never ``FALSE``.
* Safety fields are only ``SignalValue.TRUE`` (danger cue) or
  ``SignalValue.UNKNOWN`` (no cue). Never ``FALSE``.
* The ``FALSE`` signal value is never produced anywhere in this module.

Because every safety field is at most ``UNKNOWN``, the normalizer maps each to
descriptor ``True`` and the router returns R0 / None. Therefore a
``RequestSignals`` derived purely from this extractor can never route to a
wrapper on its own -- confident safety absence must come from a separate,
explicitly-authorized source, never from the text.
"""

from __future__ import annotations

import re

from agent.subagent_task_descriptor import RequestRef, RequestSignals, SignalValue


# --- field name lists (mirror RequestSignals; kept local, not imported) -----

_INTENT_FIELDS = (
    "public_url_present",
    "open_web_search",
    "local_file_ref_present",
    "candidate_change_present",
    "judge_requested",
    "test_plan_requested",
    "symptom_present",
    "broad_inspection_requested",
    "gate_question_present",
    "gate_execute_requested",
)

_SAFETY_FIELDS = (
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


def _c(*patterns: str) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


# --- intent cues (high precision; absence -> UNKNOWN) -----------------------

_INTENT_CUES: dict[str, tuple[re.Pattern, ...]] = {
    "public_url_present": _c(r"https?://", r"\bpublic url\b"),
    "open_web_search": _c(
        r"\bsearch (?:the )?(?:web|online|internet)\b",
        r"\bweb search\b",
        r"\blook up\b.*\bonline\b",
    ),
    "local_file_ref_present": _c(
        r"\b(?:read|inspect|review)\b.*\bfile\b",
        r"\bin (?:the )?(?:file|repo|module)\b",
    ),
    "candidate_change_present": _c(
        r"\bdiff\b", r"\bpatch\b", r"\bcommit\b",
        r"\bchanged[- ]file", r"\bpull request\b",
    ),
    "judge_requested": _c(
        r"\breview this\b", r"\bsafe to (?:merge|push)\b",
        r"\bjudge\b.*\b(?:risk|change|patch)\b", r"\bis this (?:diff|patch|commit)\b",
    ),
    "test_plan_requested": _c(
        r"\btest plan\b", r"\bsmoke plan\b", r"\bpass criteria\b",
        r"\bstop conditions?\b", r"\b(?:design|write) (?:targeted )?tests?\b",
        r"\bcoverage\b",
    ),
    "symptom_present": _c(
        r"\btraceback\b", r"\bregression\b", r"\bfailing test\b",
        r"\bstack ?trace\b", r"\bwhy (?:is|does).*\bfail",
    ),
    "broad_inspection_requested": _c(
        r"\baudit\b", r"\binspect\b.*\b(?:boundary|surface|exposure)\b",
        r"\bcheck\b.*\b(?:static )?(?:safety )?surface\b",
        r"\broute exposure\b", r"\btoolset boundary\b",
    ),
    "gate_question_present": _c(
        r"\b(?:request|proceed to|advance).*\bgate\b",
        r"\bauthorization gate\b", r"\bmay we (?:request|proceed)\b",
        r"\bgate (?:remain|stay) blocked\b",
    ),
    "gate_execute_requested": _c(
        r"\bexecute the gate\b", r"\brun the gate\b", r"\bopen the gate\b",
    ),
}


# --- safety danger cues (any match -> TRUE; absence -> UNKNOWN) -------------

_SAFETY_CUES: dict[str, tuple[re.Pattern, ...]] = {
    "mutation_requested": _c(
        r"\b(?:edit|modify|apply|write|patch|change|fix it|rewrite)\b",
        r"\bapply the (?:fix|patch)\b",
    ),
    "execution_requested": _c(
        r"\brun\b", r"\bexecute\b", r"\bpytest\b", r"\b(?:run|execute) the (?:tests?|smoke)\b",
    ),
    "runtime_start_requested": _c(
        r"\bstart (?:the )?(?:hermes )?runtime\b", r"\blaunch (?:the )?(?:agent|runtime)\b",
        r"\bstart the agent loop\b",
    ),
    "real_delegate_requested": _c(
        r"\bdelegate_task\b", r"\bspawn (?:a )?child agent\b",
        r"\breal delegat", r"\bfan out\b",
    ),
    "secret_or_forbidden_path_referenced": _c(
        r"\.env\b", r"\bcredential", r"\bsecret", r"\btoken\b",
        r"\bapi[_-]?key\b", r"/opt/data", r"\buser\.md\b", r"\bmemory\.md\b",
        r"\bconfig\.yaml\b",
    ),
    "external_surface_requested": _c(
        r"\bgateway\b", r"\brpc\b", r"\btui\b", r"\bexpose\b.*\broute\b",
        r"\bmodel[- ]callable\b",
    ),
    "external_connect_requested": _c(
        r"\btelegram\b", r"\bfeishu\b", r"\bdiscord\b", r"\bwebhook\b",
        r"\bconnect (?:to )?(?:an )?external\b",
    ),
    "git_mutation_requested": _c(
        r"\bpush\b", r"\bcommit\b", r"\bgit (?:reset|checkout|rebase|pull|fetch)\b",
        r"\bforce[- ]push\b", r"\bcreate (?:a )?pr\b", r"\bpull request\b",
    ),
    "env_mutation_requested": _c(
        r"\bpip install\b", r"\bcreate (?:a )?venv\b", r"\binstall (?:the )?deps",
        r"\binstall dependenc", r"\bapt install\b",
    ),
}


# --- ref shape detectors ----------------------------------------------------

_URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)

# A local path-like token: an absolute/relative POSIX path or a Windows path.
# The lookbehind excludes a preceding word char, ':' or '/', so the "//host"
# portion of a URL is not misread as a local path.
_LOCAL_PATH_RE = re.compile(
    r"(?<![\w:/])(?:~?/[\w.\-/]+|\.{1,2}/[\w.\-/]+|[A-Za-z]:\\[\w.\\\-]+)"
)

# Forbidden-looking reference patterns (inclusion-biased).
_FORBIDDEN_REF_RE = re.compile(
    r"(?:\.env\b|/opt/data|\buser\.md\b|\bmemory\.md\b|\bconfig\.yaml\b"
    r"|\bcredential|\bsecret|\btoken\b|\bapi[_-]?key\b)",
    re.IGNORECASE,
)


# --- helpers (pure) ---------------------------------------------------------

def _match_any(text: str, patterns: tuple[re.Pattern, ...]) -> bool:
    return any(p.search(text) for p in patterns)


def _intent_value(text: str, field: str) -> SignalValue:
    """TRUE on an explicit cue, else UNKNOWN. Never FALSE."""
    if _match_any(text, _INTENT_CUES.get(field, ())):
        return SignalValue.TRUE
    return SignalValue.UNKNOWN


def _safety_value(text: str, field: str) -> SignalValue:
    """TRUE on a danger cue, else UNKNOWN. Never FALSE."""
    if _match_any(text, _SAFETY_CUES.get(field, ())):
        return SignalValue.TRUE
    return SignalValue.UNKNOWN


def _is_forbidden_ref(token: str) -> bool:
    return bool(_FORBIDDEN_REF_RE.search(token))


def _extract_refs(text: str) -> tuple[list[RequestRef], bool]:
    """Classify ref-shaped tokens by string shape only (never dereferenced).

    Returns (refs, any_forbidden). Forbidden detection runs before generic
    local-path classification so a forbidden path is tagged forbidden=True.
    """
    refs: list[RequestRef] = []
    any_forbidden = False
    seen: set[str] = set()

    for m in _URL_RE.finditer(text):
        tok = m.group(0)
        if tok in seen:
            continue
        seen.add(tok)
        if _is_forbidden_ref(tok):
            refs.append(RequestRef(type="public_url", ref=tok, forbidden=True))
            any_forbidden = True
        else:
            refs.append(RequestRef(type="public_url", ref=tok))

    for m in _LOCAL_PATH_RE.finditer(text):
        tok = m.group(0)
        if tok in seen:
            continue
        seen.add(tok)
        if _is_forbidden_ref(tok):
            refs.append(RequestRef(type="allowlisted_file", ref=tok, forbidden=True))
            any_forbidden = True
        else:
            refs.append(RequestRef(type="allowlisted_file", ref=tok))

    return refs, any_forbidden


# --- public entry point -----------------------------------------------------

def extract_request_signals_from_text(text: str) -> RequestSignals:
    """Extract one ``RequestSignals`` proposal from raw, untrusted text.

    Pure and deterministic. Produces only ``TRUE``/``UNKNOWN`` signals (never
    ``FALSE``); refs are classified by shape and never dereferenced. The
    original text is copied verbatim into ``declared_intent`` for provenance
    only -- it is non-authoritative and drives no signal.
    """
    # Defensive coercion: non-str -> empty (all-UNKNOWN -> R0 downstream).
    if not isinstance(text, str):
        text = ""

    intent = {f: _intent_value(text, f) for f in _INTENT_FIELDS}
    safety = {f: _safety_value(text, f) for f in _SAFETY_FIELDS}

    refs, any_forbidden = _extract_refs(text)

    # Refs of a given shape raise the corresponding intent signal.
    if any(r.type == "public_url" for r in refs):
        intent["public_url_present"] = SignalValue.TRUE
    if any(r.type == "allowlisted_file" for r in refs):
        intent["local_file_ref_present"] = SignalValue.TRUE

    # A forbidden ref escalates the secret/forbidden-path safety flag.
    if any_forbidden:
        safety["secret_or_forbidden_path_referenced"] = SignalValue.TRUE

    return RequestSignals(
        **intent,
        **safety,
        declared_refs=refs,
        declared_intent=text,
    )
