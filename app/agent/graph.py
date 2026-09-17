"""LangGraph workflow for the maintenance agent.

The pipeline is fully deterministic in its default configuration. No LLM is
involved unless the LLM planner is explicitly enabled, and the RAG tool reaches
the knowledge base through a retrieval-only provider.

Node order:

* ``route_query`` runs the rule-based parser and fills the query entities.
* ``plan_actions`` is the planner node: it selects the planner configured by
  ``PLANNER_MODE`` and emits the plan. The frozen rule planner defined in this
  module is the baseline; the LLM planner is additive and never replaces it.
* ``execute_tools`` is the executor: it runs exactly the planned tools off the
  registry and never re-decides what to call. It calls no LLM.
* ``retrieve_context`` standardizes the tool results into ``Evidence`` records.
  It calls no external system and is pure post-processing.
* ``synthesize`` renders the Chinese answer, including the maintenance manual
  section, using only facts the tools actually returned. No LLM is used to write
  the answer.
"""

from collections.abc import Callable
from time import perf_counter
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.agent.parser import Intent, parse_query
from app.agent.planners.dispatch import dispatch_plan
from app.agent.state import MaintenanceState
from app.observability import instrumentation
from app.schemas.evidence import Evidence, SourceType
from app.tools import registry
from app.tools.arguments import ToolArgumentError, validate_tool_arguments
from app.tools.maintenance_manual_tool import MANUAL_TOOL_SOURCE
from app.tools.names import ToolName

DEVICE_TOOL_SOURCE = "sqlite:devices"
ALARM_TOOL_SOURCE = "data/alarms.json"

# Intents that justify searching the maintenance manual.
MANUAL_SEARCH_INTENTS: frozenset[str] = frozenset(
    {Intent.ALARM_DIAGNOSIS.value, Intent.MAINTENANCE_ADVICE.value}
)


def route_query(state: MaintenanceState) -> dict[str, Any]:
    """Parse the query and fill the entity and intent fields.

    Existing ``equipment_id`` / ``alarm_code`` values are preserved when the
    parser cannot resolve them, so a caller-supplied device id is never dropped.
    """
    parsed = parse_query(state.get("query", ""))

    update: dict[str, Any] = {
        "intent": parsed.intent.value,
        "next_step": "plan_actions",
    }
    if parsed.equipment_id:
        update["equipment_id"] = parsed.equipment_id
    if parsed.alarm_code:
        update["alarm_code"] = parsed.alarm_code
    return update


def plan_actions(state: MaintenanceState) -> dict[str, Any]:
    """Decide which tools the query needs (the frozen rule planner).

    FROZEN BASELINE. The body of this function is unchanged since V0.4 and is the
    default planner (``PLANNER_MODE=rule``). V0.5 adds an LLM planner alongside
    it; it does not replace or modify these rules.

    ``get_device_status`` is required when a device is known,
    ``query_alarm_code`` when an alarm code is known, and
    ``search_maintenance_manual`` when the intent calls for diagnosis or
    maintenance guidance. This is the only place the deterministic selection
    rules live, and it is the only place the rule planner selects tools.
    """
    required_tools: list[str] = []
    if state.get("equipment_id"):
        required_tools.append(ToolName.GET_DEVICE_STATUS.value)
    if state.get("alarm_code"):
        required_tools.append(ToolName.QUERY_ALARM_CODE.value)
    if state.get("intent") in MANUAL_SEARCH_INTENTS:
        required_tools.append(ToolName.SEARCH_MAINTENANCE_MANUAL.value)
    return {"required_tools": required_tools}


def _device_status_arguments(state: MaintenanceState) -> dict[str, Any]:
    """Map agent state onto ``get_device_status`` arguments."""
    return {"device_id": state.get("equipment_id")}


def _alarm_code_arguments(state: MaintenanceState) -> dict[str, Any]:
    """Map agent state onto ``query_alarm_code`` arguments."""
    return {"alarm_code": state.get("alarm_code")}


def _manual_query(state: MaintenanceState) -> str:
    """Build the manual search query from the state.

    The raw user query is preferred. When the caller invoked the graph with
    entities only, the known equipment id and alarm code are joined into a
    deterministic keyword query rather than searching with an empty string.
    """
    query = (state.get("query") or "").strip()
    if query:
        return query
    return " ".join(
        str(value) for value in (state.get("equipment_id"), state.get("alarm_code")) if value
    )


def _manual_search_arguments(state: MaintenanceState) -> dict[str, Any]:
    """Map agent state onto ``search_maintenance_manual`` arguments."""
    return {"query": _manual_query(state)}


ArgumentBuilder = Callable[[MaintenanceState], dict[str, Any]]

# Argument adapters live in the agent layer on purpose: the tools stay ignorant
# of the state shape, and the executor only translates state into call
# arguments without choosing anything.
_TOOL_ARGUMENTS: dict[str, ArgumentBuilder] = {
    ToolName.GET_DEVICE_STATUS.value: _device_status_arguments,
    ToolName.QUERY_ALARM_CODE.value: _alarm_code_arguments,
    ToolName.SEARCH_MAINTENANCE_MANUAL.value: _manual_search_arguments,
}


def build_tool_arguments(name: str, state: MaintenanceState) -> dict[str, Any] | None:
    """Return the arguments the executor derives for ``name`` from ``state``.

    A read-only accessor over the frozen adapter table above. It exists so the
    evaluation harness can score argument accuracy for the rule planner, which
    keeps its arguments out of the plan and derives them here at execution time.
    Exposing the mapping is what prevents a second, drifting copy in the harness.

    It selects nothing and changes no behaviour. Returns ``None`` for a tool with
    no adapter.
    """
    builder = _TOOL_ARGUMENTS.get(name)
    if builder is None:
        return None
    return builder(state)


def _as_payload(value: Any) -> Any:
    """Convert a tool return value into a JSON-serializable payload."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _invoke_tool(name: str, spec: Any, arguments: dict[str, Any]) -> Any:
    """Call one registered tool, measured.

    The call itself is unchanged: the same callable, the same keyword arguments,
    the same return value. This wrapper adds a duration, a span and an outcome
    derived from the tool's own payload, so the executor's decisions and the tool
    contract are untouched.

    An exception propagates exactly as before. A tool that raises is a tool
    failure, and swallowing it here would turn it into an empty result the caller
    could mistake for "nothing found".
    """
    started = perf_counter()
    instrumentation.tool_started(tool=name)
    with instrumentation.tool_span(tool=name) as active_span:
        try:
            value = spec.func(**arguments)
        except Exception as exc:
            instrumentation.tool_failed(
                tool=name,
                error_type=type(exc).__name__,
                duration_ms=instrumentation.elapsed_ms(started),
            )
            active_span.record_error(type(exc).__name__)
            raise

        payload = _as_payload(value)
        status = instrumentation.tool_status_from_payload(payload)
        instrumentation.tool_completed(
            tool=name,
            status=status,
            duration_ms=instrumentation.elapsed_ms(started),
        )
        active_span.set_attribute("status", status)
        return payload


def _execute_planned_calls(planned: list[dict[str, Any]]) -> dict[str, Any]:
    """Run an explicit call list produced by the LLM planner.

    The executor still performs no selection and no repair. Arguments are
    validated against the tool's own input model, and a call whose arguments do
    not validate is reported and skipped rather than completed by guessing the
    missing values.
    """
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for call in planned:
        if not isinstance(call, dict):
            errors.append("malformed planned call: expected an object")
            continue

        name = call.get("tool")
        if not isinstance(name, str) or not name:
            errors.append("malformed planned call: missing tool name")
            continue

        try:
            spec = registry.get(name)
        except KeyError:
            errors.append(f"tool not registered: {name}")
            continue

        arguments = call.get("arguments")
        if arguments is None:
            arguments = {}
        try:
            validated = validate_tool_arguments(name, arguments)
        except ToolArgumentError as exc:
            errors.append(str(exc))
            continue

        results.append({"tool": name, "result": _invoke_tool(name, spec, validated)})

    update: dict[str, Any] = {"tool_results": results}
    if errors:
        update["error"] = "; ".join(errors)
    return update


def execute_tools(state: MaintenanceState) -> dict[str, Any]:
    """Run exactly the tools the planner requested (the executor).

    The executor performs no tool selection and calls no LLM. It reads the plan
    and resolves each name off the shared registry:

    * when the planner produced an explicit call list (the LLM planner), those
      calls run with the planner's arguments, validated against the tool's own
      input model;
    * otherwise the arguments are derived deterministically from the state, which
      is the frozen rule behaviour;

    Unknown names, or arguments that do not validate, are reported through
    ``error`` rather than silently substituted or completed.
    """
    planned_calls = state.get("planned_tool_calls")
    if planned_calls:
        return _execute_planned_calls(planned_calls)

    required_tools = state.get("required_tools") or []
    if not required_tools:
        return {"tool_results": []}

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for name in required_tools:
        try:
            spec = registry.get(name)
        except KeyError:
            errors.append(f"tool not registered: {name}")
            continue

        build_arguments = _TOOL_ARGUMENTS.get(name)
        if build_arguments is None:
            errors.append(f"no argument mapping for tool: {name}")
            continue

        arguments = build_arguments(state)
        missing = [key for key, value in arguments.items() if value is None or value == ""]
        if missing:
            errors.append(f"missing state values for {name}: {', '.join(missing)}")
            continue

        results.append({"tool": name, "result": _invoke_tool(name, spec, arguments)})

    update: dict[str, Any] = {"tool_results": results}
    if errors:
        update["error"] = "; ".join(errors)
    return update


def _find_result(
    tool_results: list[dict[str, Any]],
    tool_name: str,
) -> dict[str, Any] | None:
    """Return the payload of the first result produced by ``tool_name``."""
    for entry in tool_results:
        if entry.get("tool") == tool_name:
            payload = entry.get("result")
            if isinstance(payload, dict):
                return payload
    return None


def _device_label(payload: dict[str, Any]) -> str:
    device_id = str(payload.get("device_id", ""))
    device_name = payload.get("device_name")
    return f"{device_id}（{device_name}）" if device_name else device_id


def _device_facts(payload: dict[str, Any]) -> list[str]:
    """Return the fact strings the device tool actually returned.

    Only fields the payload actually carries become facts. An external source
    contributes its own readings under their own names, so nothing is projected
    onto a PLC-shaped field it does not belong in.
    """
    facts: list[str] = []
    if payload.get("data_source"):
        facts.append(f"数据来源 {payload['data_source']}")
    if payload.get("status"):
        facts.append(f"状态 {payload['status']}")
    if payload.get("temperature") is not None:
        facts.append(f"温度 {payload['temperature']}")
    if payload.get("pressure") is not None:
        facts.append(f"压力 {payload['pressure']}")
    if payload.get("rpm") is not None:
        facts.append(f"转速 {payload['rpm']}")
    if payload.get("alarm_code"):
        facts.append(f"报警码 {payload['alarm_code']}")
    measurements = payload.get("measurements") or {}
    if measurements:
        facts.append(
            "实测读数 " + "，".join(f"{key}={value}" for key, value in measurements.items())
        )
    if payload.get("source_timestamp"):
        facts.append(f"数据时间戳 {payload['source_timestamp']}")
    return facts


def _device_evidence(payload: dict[str, Any]) -> Evidence:
    device_id = str(payload.get("device_id", ""))
    if payload.get("error"):
        content = f"设备 {device_id} 状态不可用：{payload['error']}"
    elif not payload.get("found"):
        content = f"设备 {device_id} 未找到：数据库中没有该设备记录。"
    else:
        facts = _device_facts(payload)
        detail = "，".join(facts) if facts else "无可用状态字段"
        content = f"设备 {_device_label(payload)}：{detail}。"

    return Evidence(
        source_type=SourceType.TOOL,
        # A fact is attributed to the source that produced it. External sources
        # name themselves, so evidence never claims to come from the seeded
        # database when it did not.
        source=str(payload.get("data_source") or DEVICE_TOOL_SOURCE),
        tool_name=ToolName.GET_DEVICE_STATUS.value,
        content=content,
    )


def _alarm_evidence(payload: dict[str, Any]) -> Evidence:
    alarm_code = str(payload.get("alarm_code", ""))
    if not payload.get("found"):
        content = f"报警码 {alarm_code} 未找到：报警目录中没有该编码。"
    else:
        name = payload.get("name") or "未命名"
        severity = payload.get("severity") or "未提供"
        category = payload.get("category") or "未分类"
        content = f"报警码 {alarm_code}（{name}）：严重等级 {severity}，类别 {category}。"

    return Evidence(
        source_type=SourceType.TOOL,
        source=ALARM_TOOL_SOURCE,
        tool_name=ToolName.QUERY_ALARM_CODE.value,
        content=content,
    )


def _manual_evidence(payload: dict[str, Any]) -> list[Evidence]:
    """Turn manual retrieval hits into document evidence.

    Only actual hits become evidence. A retrieval miss, or a retrieval that
    could not run, produces no document evidence: the reason is reported in the
    answer instead of being dressed up as a citable fact.
    """
    if not payload.get("found"):
        return []

    evidence: list[Evidence] = []
    for hit in payload.get("results") or []:
        content = hit.get("content")
        if not content:
            continue
        evidence.append(
            Evidence(
                source_type=SourceType.DOCUMENT,
                source=str(hit.get("document") or MANUAL_TOOL_SOURCE),
                tool_name=ToolName.SEARCH_MAINTENANCE_MANUAL.value,
                content=str(content),
                document=hit.get("document"),
                page=hit.get("page"),
                score=hit.get("score"),
                score_semantics=hit.get("score_semantics"),
                higher_is_better=hit.get("higher_is_better"),
            )
        )
    return evidence


def _standardize_evidence(tool_results: list[dict[str, Any]]) -> list[Evidence]:
    """Standardize raw tool results into evidence records.

    Pure post-processing: it reads what the executor already ran and never
    triggers a call of its own.
    """
    evidence: list[Evidence] = []
    for entry in tool_results:
        tool_name = entry.get("tool")
        payload = entry.get("result")
        if not isinstance(payload, dict):
            continue
        if tool_name == ToolName.GET_DEVICE_STATUS.value:
            evidence.append(_device_evidence(payload))
        elif tool_name == ToolName.QUERY_ALARM_CODE.value:
            evidence.append(_alarm_evidence(payload))
        elif tool_name == ToolName.SEARCH_MAINTENANCE_MANUAL.value:
            evidence.extend(_manual_evidence(payload))
    return evidence


def retrieve_context(state: MaintenanceState) -> dict[str, Any]:
    """Standardize tool results into evidence records.

    This node performs no retrieval of its own and contacts no external
    system. Retrieval already happened in ``execute_tools``; this step exists so
    that evidence construction is a single, testable place in the pipeline.
    """
    tool_results = state.get("tool_results") or []
    evidence = _standardize_evidence(tool_results)
    return {"retrieved_context": [item.model_dump(mode="json") for item in evidence]}


def _device_section(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "【设备状态】\n本次未查询设备状态。"

    device_id = str(payload.get("device_id", ""))
    if payload.get("error"):
        return f"【设备状态】\n设备 {device_id} 状态不可用：{payload['error']}"
    if not payload.get("found"):
        return f"【设备状态】\n设备 {device_id} 未找到：数据库中没有该设备记录。"

    lines = [f"设备编号与名称：{_device_label(payload)}"]
    if payload.get("device_type"):
        lines.append(f"设备类型：{payload['device_type']}")
    if payload.get("location"):
        lines.append(f"安装位置：{payload['location']}")

    status_line = f"当前状态：{payload.get('status') or '未提供'}"
    # A derived label carries its basis with it, so a reader can never mistake it
    # for a state the source published.
    if payload.get("status_derivation"):
        status_line += f"（{payload['status_derivation']}）"
    lines.append(status_line)

    metrics: list[str] = []
    if payload.get("temperature") is not None:
        metrics.append(f"温度 {payload['temperature']}")
    if payload.get("pressure") is not None:
        metrics.append(f"压力 {payload['pressure']}")
    if payload.get("rpm") is not None:
        metrics.append(f"转速 {payload['rpm']}")

    measurements = payload.get("measurements") or {}
    if metrics:
        lines.append("关键状态：" + "，".join(metrics))
    elif not measurements:
        lines.append("关键状态：无可用测量值")

    if payload.get("data_source"):
        lines.append(f"数据来源：{payload['data_source']}")
    if payload.get("source_timestamp"):
        lines.append(f"数据时间戳：{payload['source_timestamp']}")

    if measurements:
        lines.append("实测读数：")
        lines.extend(f"- {key} = {value}" for key, value in measurements.items())

    signals = payload.get("digital_signals") or {}
    if signals:
        lines.append("数字信号：")
        lines.extend(f"- {key} = {value}" for key, value in signals.items())

    lines.append(f"报警码：{payload.get('alarm_code') or '无'}")
    if payload.get("last_maintenance_time"):
        lines.append(f"最近保养时间：{payload['last_maintenance_time']}")
    return "【设备状态】\n" + "\n".join(lines)


def _alarm_section(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "【报警信息】\n本次未查询报警信息。"

    alarm_code = str(payload.get("alarm_code", ""))
    if not payload.get("found"):
        return f"【报警信息】\n报警码 {alarm_code} 未找到：报警目录中没有该编码。"

    lines = [f"报警码：{alarm_code}"]
    if payload.get("name"):
        lines.append(f"报警名称：{payload['name']}")
    if payload.get("description"):
        lines.append(f"报警说明：{payload['description']}")
    lines.append(f"报警严重等级：{payload.get('severity') or '未提供'}")
    if payload.get("category"):
        lines.append(f"报警类别：{payload['category']}")

    causes = payload.get("possible_causes") or []
    if causes:
        lines.append("可能原因：\n" + "\n".join(f"- {cause}" for cause in causes))
    else:
        lines.append("可能原因：报警目录中未提供。")

    actions = payload.get("recommended_actions") or []
    if actions:
        lines.append("建议动作：\n" + "\n".join(f"- {action}" for action in actions))
    else:
        lines.append("建议动作：报警目录中未提供。")

    return "【报警信息】\n" + "\n".join(lines)


def _manual_section(payload: dict[str, Any] | None) -> str:
    """Render the maintenance manual section.

    The three outcomes stay distinguishable: not requested, requested but
    unavailable, and requested and completed with either no hits or some hits.
    """
    header = "【维护手册证据】"

    if payload is None:
        return f"{header}\n本次未检索维护手册。"

    if payload.get("error"):
        return f"{header}\n维护手册检索不可用：{payload['error']}"

    results = payload.get("results") or []
    if not payload.get("found") or not results:
        query = payload.get("query", "")
        return f"{header}\n未在维护手册中检索到相关片段（检索词：{query}）。"

    provider = payload.get("provider") or "unknown"
    mode = payload.get("retrieval_mode")
    summary = f"命中 {len(results)} 条片段（provider：{provider}"
    if mode:
        summary += f"，检索模式：{mode}"
    summary += "）。"

    lines = [header, summary]
    for index, hit in enumerate(results, start=1):
        origin: list[str] = []
        if hit.get("document"):
            origin.append(str(hit["document"]))
        if hit.get("page") is not None:
            origin.append(f"第 {hit['page']} 页")
        if hit.get("section"):
            origin.append(f"章节 {hit['section']}")
        if hit.get("chunk_id"):
            origin.append(f"chunk {hit['chunk_id']}")
        if hit.get("score") is not None:
            label = f"相关度 {hit['score']}"
            semantics = hit.get("score_semantics")
            direction = hit.get("higher_is_better")
            if semantics:
                label += f"（{semantics}，{'越大' if direction else '越小'}越相关）"
            origin.append(label)
        label = " | ".join(origin) if origin else "来源字段未提供"
        lines.append(f"- [{index}] {label}")
        lines.append(f"  正文摘录：{hit.get('content', '')}")

    return "\n".join(lines)


def synthesize(state: MaintenanceState) -> dict[str, Any]:
    """Turn tool results into evidence and a deterministic Chinese summary.

    Only facts present in ``tool_results`` are used. Nothing is inferred or
    invented: when a lookup misses, the answer says so explicitly and omits the
    fields that would otherwise be fabricated.

    Evidence is taken from ``retrieved_context`` when the ``retrieve_context``
    node has already standardized it. Calling this node standalone still works
    and standardizes on the spot.
    """
    tool_results = state.get("tool_results") or []
    device_payload = _find_result(tool_results, ToolName.GET_DEVICE_STATUS.value)
    alarm_payload = _find_result(tool_results, ToolName.QUERY_ALARM_CODE.value)
    manual_payload = _find_result(tool_results, ToolName.SEARCH_MAINTENANCE_MANUAL.value)

    evidence = state.get("retrieved_context")
    if not evidence:
        evidence = [item.model_dump(mode="json") for item in _standardize_evidence(tool_results)]

    # The span carries the evidence count and nothing else: the answer text is
    # generated here and must not become a span attribute.
    with instrumentation.synthesis_span(evidence_count=len(evidence)):
        final_answer = "\n\n".join(
            [
                _device_section(device_payload),
                _alarm_section(alarm_payload),
                _manual_section(manual_payload),
            ]
        )

    return {
        "evidence": evidence,
        "final_answer": final_answer,
        # ``answer`` mirrors ``final_answer`` for the legacy response model.
        "answer": final_answer,
    }


def select_and_plan(state: MaintenanceState) -> dict[str, Any]:
    """Planner node: dispatch to the configured planner.

    The graph node keeps its existing name and position. Only the choice of
    planner is configurable; the frozen rule planner is passed in as the
    baseline, so this module stays the single owner of the deterministic rules.
    """
    return dispatch_plan(state, rule_planner=plan_actions)


def build_graph():
    """Build and compile the maintenance workflow graph."""
    graph = StateGraph(MaintenanceState)

    graph.add_node("route_query", route_query)
    graph.add_node("plan_actions", select_and_plan)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "route_query")
    graph.add_edge("route_query", "plan_actions")
    graph.add_edge("plan_actions", "execute_tools")
    graph.add_edge("execute_tools", "retrieve_context")
    graph.add_edge("retrieve_context", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    """Return a lazily compiled singleton graph instance."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
