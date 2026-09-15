"""Tests for the rule-based query parser and the state-aware agent nodes."""

from app.agent.graph import plan_actions, route_query
from app.agent.parser import Intent, ParsedQuery, parse_query
from app.agent.state import MaintenanceState
from app.tools import registry
from app.tools.names import ToolName


def test_parsed_query_exposes_exactly_three_fields() -> None:
    parsed = parse_query("PLC-001")

    assert isinstance(parsed, ParsedQuery)
    assert set(parsed.model_dump()) == {"equipment_id", "alarm_code", "intent"}


def test_parse_chinese_fault_description() -> None:
    parsed = parse_query("包装线PLC报警F0045怎么办")

    assert parsed.equipment_id == "PLC-001"
    assert parsed.alarm_code == "F0045"
    assert parsed.intent is Intent.ALARM_DIAGNOSIS


def test_parse_device_id_formats() -> None:
    cases = [
        ("PLC-001", "PLC-001"),
        ("plc-001 温度多少", "PLC-001"),
        ("PLC-002 压力异常", "PLC-002"),
        ("Robot-001 状态", "Robot-001"),
        ("robot-001", "Robot-001"),
        ("CNC-001 需要保养", "CNC-001"),
    ]

    for text, expected in cases:
        assert parse_query(text).equipment_id == expected, text


def test_parse_device_name_aliases() -> None:
    assert parse_query("焊接机器人报警了").equipment_id == "Robot-001"
    assert parse_query("码垛机器人运行正常吗").equipment_id == "Robot-002"
    assert parse_query("灌装线PLC 压力如何").equipment_id == "PLC-002"


def test_parse_unknown_device_id_is_kept_verbatim() -> None:
    # Unknown identifiers are returned so the device tool can report found=False.
    assert parse_query("PLC-999 什么状态").equipment_id == "PLC-999"


def test_parse_alarm_code_formats() -> None:
    cases = [
        ("F0045", "F0045"),
        ("f0045 是什么报警", "F0045"),
        ("F0112 怎么处理", "F0112"),
        ("E0071", "E0071"),
        ("报警码 W0100", "W0100"),
    ]

    for text, expected in cases:
        assert parse_query(text).alarm_code == expected, text


def test_parse_alarm_code_is_not_confused_with_device_id() -> None:
    parsed = parse_query("PLC-001")

    assert parsed.alarm_code is None
    assert parsed.equipment_id == "PLC-001"


def test_parse_intent_detection() -> None:
    assert parse_query("PLC-001").intent is Intent.DEVICE_STATUS
    assert parse_query("PLC-001 温度多少").intent is Intent.DEVICE_STATUS
    assert parse_query("PLC-001 怎么办").intent is Intent.MAINTENANCE_ADVICE
    assert parse_query("报警了").intent is Intent.ALARM_DIAGNOSIS


def test_parse_blank_input_returns_unknown() -> None:
    parsed = parse_query("   ")

    assert parsed.equipment_id is None
    assert parsed.alarm_code is None
    assert parsed.intent is Intent.UNKNOWN


def test_route_query_fills_equipment_and_alarm() -> None:
    update = route_query(MaintenanceState(query="包装线PLC报警F0045怎么办"))

    assert update["equipment_id"] == "PLC-001"
    assert update["alarm_code"] == "F0045"
    assert update["intent"] == "alarm_diagnosis"
    assert update["next_step"] == "plan_actions"


def test_route_query_preserves_caller_supplied_equipment_id() -> None:
    update = route_query(MaintenanceState(query="现在什么状态", equipment_id="PLC-002"))

    # The parser finds no device in the text, so the existing value survives.
    assert "equipment_id" not in update
    assert update["intent"] == "device_status"


def test_plan_actions_returns_both_tools() -> None:
    update = plan_actions(MaintenanceState(equipment_id="PLC-001", alarm_code="F0045"))

    assert update["required_tools"] == [
        ToolName.GET_DEVICE_STATUS.value,
        ToolName.QUERY_ALARM_CODE.value,
    ]


def test_plan_actions_with_device_only() -> None:
    update = plan_actions(MaintenanceState(equipment_id="PLC-001"))

    assert update["required_tools"] == [ToolName.GET_DEVICE_STATUS.value]


def test_plan_actions_with_alarm_only() -> None:
    update = plan_actions(MaintenanceState(alarm_code="F0045"))

    assert update["required_tools"] == [ToolName.QUERY_ALARM_CODE.value]


def test_plan_actions_only_emits_registry_names() -> None:
    update = plan_actions(MaintenanceState(equipment_id="PLC-001", alarm_code="F0045"))
    registered = set(registry.names())

    assert set(update["required_tools"]).issubset(registered)


def test_plan_actions_with_empty_state() -> None:
    assert plan_actions(MaintenanceState())["required_tools"] == []


def test_route_query_then_plan_actions_chains() -> None:
    query = "包装线PLC报警F0045怎么办"
    routed = route_query(MaintenanceState(query=query))

    state = MaintenanceState(
        query=query,
        equipment_id=routed.get("equipment_id"),
        alarm_code=routed.get("alarm_code"),
        intent=routed.get("intent"),
    )

    assert plan_actions(state)["required_tools"] == [
        ToolName.GET_DEVICE_STATUS.value,
        ToolName.QUERY_ALARM_CODE.value,
        ToolName.SEARCH_MAINTENANCE_MANUAL.value,
    ]


def test_state_accepts_v03_fields() -> None:
    state = MaintenanceState(
        intent="alarm_diagnosis",
        required_tools=["device_tool", "alarm_tool"],
        evidence=[{"tool": "get_device_status"}],
        final_answer="",
    )

    assert state["intent"] == "alarm_diagnosis"
    assert state["evidence"] == [{"tool": "get_device_status"}]
