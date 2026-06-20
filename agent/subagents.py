"""Restricted named sub-agent routing helpers.

MVP-1 scope:
- internal-only wrapper
- exactly @research and @audit_static
- leaf-only delegation
- fixed toolsets
- fail-closed validation
- no gateway/TUI/RPC/model-callable activation

MVP-2 Batch 1 additions (internal-only, same constraints):
- @debug_static, @test_planner, @patch_reviewer
- all three share the read-only static_analysis_leaf toolset

MVP-2 @security_gate (internal-only, same constraints):
- @security_gate is a read-only gate-decision reviewer that inspects
  available evidence and recommends whether the next Central authorization
  gate may be requested. It is advisory only and never executes the next
  gate. It reuses the read-only static_analysis_leaf toolset (no
  delegate_task / write / execute capability).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.file_safety import get_read_block_error
from tools.path_security import has_traversal_component, validate_within_dir


STATUS_COMPLETED = "completed"
STATUS_BLOCKED = "blocked"
STATUS_FAILED = "failed"


_UNSAFE_REQUEST_FIELDS = frozenset({
    "toolsets", "role", "tasks", "max_spawn_depth", "max_concurrent_children",
    "acp_command", "acp_args", "provider", "base_url", "api_key",
    "credential", "credentials", "auth", "runtime", "runner", "ci",
})


_FORBIDDEN_BASENAMES = frozenset({
    ".env", ".env.local", ".env.development", ".env.production", ".env.test",
    ".env.staging", ".envrc", "auth.json", "auth.lock", "config.yaml",
    "webhook_subscriptions.json", ".anthropic_oauth.json", "user.md", "memory.md",
})


_FORBIDDEN_REF_FRAGMENTS = (
    "/opt/data/", "/.hermes/", "~/.hermes/", "/mcp-tokens/", "/sessions/", "/logs/",
)


_SECRET_LIKE_RE = re.compile(
    r"(api[_-]?key|secret|token|authorization|bearer\s+[a-z0-9._-]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SubagentPreset:
    name: str
    toolsets: tuple[str, ...]
    allow_public_urls: bool
    allow_file_refs: bool
    allow_report_artifacts: bool
    allow_provided_text: bool
    allowed_ref_types: frozenset[str]


RESEARCH_PRESET = SubagentPreset(
    name="@research",
    toolsets=("research_leaf",),
    allow_public_urls=True,
    allow_file_refs=False,
    allow_report_artifacts=True,
    allow_provided_text=True,
    allowed_ref_types=frozenset({"provided_text", "public_url", "report_artifact"}),
)


AUDIT_STATIC_PRESET = SubagentPreset(
    name="@audit_static",
    toolsets=("audit_static_leaf",),
    allow_public_urls=False,
    allow_file_refs=True,
    allow_report_artifacts=True,
    allow_provided_text=True,
    allowed_ref_types=frozenset({"provided_text", "allowlisted_file", "report_artifact"}),
)


DEBUG_STATIC_PRESET = SubagentPreset(
    name="@debug_static",
    toolsets=("static_analysis_leaf",),
    allow_public_urls=False,
    allow_file_refs=True,
    allow_report_artifacts=True,
    allow_provided_text=True,
    allowed_ref_types=frozenset({"provided_text", "allowlisted_file", "report_artifact"}),
)


TEST_PLANNER_PRESET = SubagentPreset(
    name="@test_planner",
    toolsets=("static_analysis_leaf",),
    allow_public_urls=False,
    allow_file_refs=True,
    allow_report_artifacts=True,
    allow_provided_text=True,
    allowed_ref_types=frozenset({"provided_text", "allowlisted_file", "report_artifact"}),
)


PATCH_REVIEWER_PRESET = SubagentPreset(
    name="@patch_reviewer",
    toolsets=("static_analysis_leaf",),
    allow_public_urls=False,
    allow_file_refs=True,
    allow_report_artifacts=True,
    allow_provided_text=True,
    allowed_ref_types=frozenset({"provided_text", "allowlisted_file", "report_artifact"}),
)


SECURITY_GATE_PRESET = SubagentPreset(
    name="@security_gate",
    toolsets=("static_analysis_leaf",),
    allow_public_urls=False,
    allow_file_refs=True,
    allow_report_artifacts=True,
    allow_provided_text=True,
    allowed_ref_types=frozenset({"provided_text", "allowlisted_file", "report_artifact"}),
)


_PRESETS = {
    RESEARCH_PRESET.name: RESEARCH_PRESET,
    AUDIT_STATIC_PRESET.name: AUDIT_STATIC_PRESET,
    DEBUG_STATIC_PRESET.name: DEBUG_STATIC_PRESET,
    TEST_PLANNER_PRESET.name: TEST_PLANNER_PRESET,
    PATCH_REVIEWER_PRESET.name: PATCH_REVIEWER_PRESET,
    SECURITY_GATE_PRESET.name: SECURITY_GATE_PRESET,
}


def _blocked(agent_name: str, reason: str) -> dict[str, Any]:
    return {
        "agent_name": agent_name or "",
        "status": STATUS_BLOCKED,
        "summary": "",
        "evidence": [],
        "limitations": [],
        "blocked_reason": reason,
        "tool_trace_summary": [],
    }


def _failed(agent_name: str, reason: str, *, summary: str = "") -> dict[str, Any]:
    return {
        "agent_name": agent_name or "",
        "status": STATUS_FAILED,
        "summary": summary or "",
        "evidence": [],
        "limitations": [],
        "blocked_reason": reason,
        "tool_trace_summary": [],
    }


def _completed(
    agent_name: str,
    summary: str,
    *,
    evidence: Sequence[dict[str, Any]] | None = None,
    limitations: Sequence[str] | None = None,
    tool_trace_summary: Sequence[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "agent_name": agent_name,
        "status": STATUS_COMPLETED,
        "summary": summary or "",
        "evidence": list(evidence or []),
        "limitations": list(limitations or []),
        "blocked_reason": None,
        "tool_trace_summary": list(tool_trace_summary or []),
    }


def _get_preset(agent_name: Any) -> SubagentPreset | None:
    if not isinstance(agent_name, str):
        return None
    return _PRESETS.get(agent_name.strip())


def _find_unsafe_fields(payload: Mapping[str, Any]) -> list[str]:
    return sorted(k for k in payload.keys() if k in _UNSAFE_REQUEST_FIELDS)


def _validate_goal(goal: Any) -> str | None:
    if not isinstance(goal, str) or not goal.strip():
        return "goal is required"
    return None


def _validate_context(context: Any) -> str | None:
    if context is None:
        return None
    if not isinstance(context, str):
        return "context must be a string"
    if _SECRET_LIKE_RE.search(context):
        return "context contains secret-like material"
    return None


def _looks_forbidden_ref(ref: str) -> bool:
    raw = str(ref or "").strip()
    lowered = raw.replace("\\", "/").lower()
    basename = Path(lowered).name

    if basename in _FORBIDDEN_BASENAMES:
        return True
    if basename.startswith(".env"):
        return True
    if basename.startswith("state.db") or basename.startswith("kanban.db"):
        return True
    if "token" in basename or "credential" in basename:
        return True

    for fragment in _FORBIDDEN_REF_FRAGMENTS:
        if fragment in lowered:
            return True

    return False


def _looks_like_local_file_ref(ref: str) -> bool:
    raw = str(ref or "").strip()
    lowered = raw.replace("\\", "/").lower()
    if lowered.startswith(("file:", "/", "~/", "./", "../")):
        return True
    if re.match(r"^[a-z]:/", lowered):
        return True
    return has_traversal_component(raw)


def _validate_allowlisted_file_ref(
    ref: str,
    allowlisted_roots: Sequence[str | Path] | None,
    index: int,
) -> str | None:
    if not allowlisted_roots:
        return f"input_refs[{index}] allowlisted_file requires allowlisted_roots"

    if has_traversal_component(ref):
        return f"input_refs[{index}] contains path traversal"

    candidate = Path(ref).expanduser()
    if not candidate.is_absolute():
        return f"input_refs[{index}] allowlisted_file must be absolute in MVP-1"

    try:
        resolved = candidate.resolve()
    except OSError as exc:
        return f"input_refs[{index}] could not be resolved: {exc}"

    read_block = get_read_block_error(str(resolved))
    if read_block:
        return f"input_refs[{index}] read blocked: {read_block}"

    for root in allowlisted_roots:
        if validate_within_dir(resolved, Path(root).expanduser()) is None:
            return None

    return f"input_refs[{index}] is outside allowlisted roots"


def _validate_one_ref(
    *,
    preset: SubagentPreset,
    item: Any,
    index: int,
    allowlisted_roots: Sequence[str | Path] | None,
) -> str | None:
    if not isinstance(item, dict):
        return f"input_refs[{index}] must be an object"

    ref_type = item.get("type")
    ref = item.get("ref")
    access = item.get("access", "read_only")

    if ref_type not in preset.allowed_ref_types:
        return f"input_refs[{index}] type is not allowed for {preset.name}"
    if access != "read_only":
        return f"input_refs[{index}] access must be read_only"
    if not isinstance(ref, str) or not ref.strip():
        return f"input_refs[{index}] ref is required"
    if _looks_forbidden_ref(ref):
        return f"input_refs[{index}] targets a forbidden path or secret-like ref"
    if _SECRET_LIKE_RE.search(ref):
        return f"input_refs[{index}] contains secret-like material"

    if ref_type == "allowlisted_file":
        if not preset.allow_file_refs:
            return f"input_refs[{index}] allowlisted_file is not allowed for {preset.name}"
        return _validate_allowlisted_file_ref(ref, allowlisted_roots, index)

    if ref_type == "public_url":
        if not preset.allow_public_urls:
            return f"input_refs[{index}] public_url is not allowed for {preset.name}"
        if not ref.startswith(("http://", "https://")):
            return f"input_refs[{index}] public_url must start with http:// or https://"

    if ref_type == "report_artifact" and _looks_like_local_file_ref(ref):
        return f"input_refs[{index}] report_artifact must not be an arbitrary local file ref"

    return None


def _validate_input_refs(
    *,
    preset: SubagentPreset,
    input_refs: Any,
    allowlisted_roots: Sequence[str | Path] | None,
) -> str | None:
    if input_refs is None:
        return None
    if not isinstance(input_refs, list):
        return "input_refs must be a list"

    for index, item in enumerate(input_refs):
        err = _validate_one_ref(
            preset=preset,
            item=item,
            index=index,
            allowlisted_roots=allowlisted_roots,
        )
        if err:
            return err
    return None


def validate_raw_named_subagent_request(
    raw: Mapping[str, Any],
    *,
    allowlisted_roots: Sequence[str | Path] | None = None,
) -> str | None:
    """Validate a raw named-subagent request envelope."""
    if not isinstance(raw, Mapping):
        return "request must be an object"

    preset = _get_preset(raw.get("agent_name"))
    if preset is None:
        return "unknown named sub-agent"

    unsafe = _find_unsafe_fields(raw)
    if unsafe:
        return f"unsafe fields are not allowed: {', '.join(unsafe)}"

    for validator in (
        lambda: _validate_goal(raw.get("goal")),
        lambda: _validate_context(raw.get("context")),
    ):
        err = validator()
        if err:
            return err

    constraints = raw.get("constraints")
    if constraints is not None and not isinstance(constraints, dict):
        return "constraints must be an object"

    expected_output = raw.get("expected_output")
    if expected_output is not None and not isinstance(expected_output, str):
        return "expected_output must be a string"

    return _validate_input_refs(
        preset=preset,
        input_refs=raw.get("input_refs"),
        allowlisted_roots=allowlisted_roots,
    )


def _build_delegate_context(
    *,
    context: str | None,
    input_refs: list[dict[str, Any]] | None,
    expected_output: str | None,
    constraints: dict[str, Any] | None,
    preset: SubagentPreset,
) -> str:
    parts = [
        f"Named sub-agent: {preset.name}",
        "Role: leaf only.",
        "Stay within the assigned tool preset.",
        "Return a concise evidence-bearing summary.",
    ]

    if context:
        parts.extend(["Context:", context])

    if input_refs:
        parts.append("Input refs:")
        for item in input_refs:
            parts.append(
                "- "
                f"type={item.get('type')} "
                f"access={item.get('access', 'read_only')} "
                f"ref={item.get('ref')}"
            )

    if expected_output:
        parts.extend(["Expected output:", expected_output])

    if constraints:
        parts.extend([
            "Constraints:",
            json.dumps(constraints, ensure_ascii=False, sort_keys=True),
        ])

    return "\n".join(parts)


def _call_delegate_task(
    *,
    goal: str,
    context: str,
    toolsets: list[str],
    role: str,
    parent_agent: Any,
) -> str:
    from tools.delegate_tool import delegate_task

    return delegate_task(
        goal=goal,
        context=context,
        toolsets=toolsets,
        tasks=None,
        max_iterations=None,
        acp_command=None,
        acp_args=None,
        role=role,
        parent_agent=parent_agent,
    )


def _summarize_tool_trace(tool_trace: Any) -> list[dict[str, str]]:
    if not isinstance(tool_trace, list):
        return []

    out: list[dict[str, str]] = []
    for item in tool_trace[:50]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("tool") or item.get("name") or "")
        status = str(item.get("status") or "")
        if name:
            out.append({"tool": name, "status": status or "unknown"})
    return out


def _normalize_delegate_result(agent_name: str, raw_result: Any) -> dict[str, Any]:
    if isinstance(raw_result, str):
        try:
            payload = json.loads(raw_result)
        except json.JSONDecodeError:
            return _failed(
                agent_name,
                "delegate_task returned non-JSON result",
                summary=raw_result[:2000],
            )
    elif isinstance(raw_result, dict):
        payload = raw_result
    else:
        return _failed(
            agent_name,
            f"delegate_task returned unsupported result type: {type(raw_result).__name__}",
        )

    if "error" in payload:
        return _failed(agent_name, str(payload.get("error") or "delegate_task error"))

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return _failed(agent_name, "delegate_task result missing results array")

    first = results[0]
    if not isinstance(first, dict):
        return _failed(agent_name, "delegate_task first result is not an object")

    status = str(first.get("status") or "").lower()
    summary = str(first.get("summary") or "")
    trace = _summarize_tool_trace(first.get("tool_trace"))

    if status == "completed":
        return _completed(agent_name, summary, tool_trace_summary=trace)

    if status in {"blocked", "interrupted", "timeout", "failed", "error"}:
        return _failed(
            agent_name,
            str(first.get("error") or status or "delegate_task failed"),
            summary=summary,
        )

    return _failed(
        agent_name,
        f"unknown delegate_task child status: {status}",
        summary=summary,
    )


def route_named_subagent(
    *,
    agent_name: str,
    goal: str,
    context: str | None = None,
    input_refs: list[dict[str, Any]] | None = None,
    expected_output: str | None = None,
    constraints: dict[str, Any] | None = None,
    parent_agent: Any,
    allowlisted_roots: Sequence[str | Path] | None = None,
    _delegate_runner: Any | None = None,
    **extra_fields: Any,
) -> dict[str, Any]:
    """Route a request to a restricted named sub-agent.

    Internal-only in MVP-1. This does not register a tool, gateway route,
    TUI route, or RPC endpoint.
    """
    raw = {
        "agent_name": agent_name,
        "goal": goal,
        "context": context,
        "input_refs": input_refs,
        "expected_output": expected_output,
        "constraints": constraints,
        **extra_fields,
    }

    preset = _get_preset(agent_name)
    if preset is None:
        return _blocked(str(agent_name or ""), "unknown named sub-agent")

    err = validate_raw_named_subagent_request(
        raw,
        allowlisted_roots=allowlisted_roots,
    )
    if err:
        return _blocked(preset.name, err)

    if parent_agent is None:
        return _blocked(preset.name, "parent_agent is required")

    delegate_context = _build_delegate_context(
        context=context,
        input_refs=input_refs,
        expected_output=expected_output,
        constraints=constraints,
        preset=preset,
    )

    runner = _delegate_runner or _call_delegate_task
    raw_result = runner(
        goal=goal.strip(),
        context=delegate_context,
        toolsets=list(preset.toolsets),
        role="leaf",
        parent_agent=parent_agent,
    )
    return _normalize_delegate_result(preset.name, raw_result)
