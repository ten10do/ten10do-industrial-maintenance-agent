"""Run one real LLM planner call, but only when a provider is actually configured.

This is the V0.5 smoke gate. It exists so the optional LLM planner can be probed
against a real endpoint without inventing a result when no endpoint is available:

* with no provider configured the script reports ``REAL_LLM_GATE_NOT_RUN``,
  prints nothing else and exits 0. It does not fabricate a plan and it does not
  call an invented host;
* with a provider configured it issues exactly one planning call and reports the
  validated plan, the provider metadata and the measured latency.

The script never prints the prompt or the raw completion. A model can echo its
prompt, and the prompt carries the tool schemas and the device vocabulary. It
never prints the API key either: only whether each configuration variable is
present.

Usage::

    python -m scripts.llm_planner_probe --query "包装线PLC报警F0045怎么办"

Run it as a module so the project root is importable. Configuration is read from
the environment or a ``.env`` file, the same way the service reads it.
``PLANNER_MODE`` is ignored here: the probe always exercises the LLM planner,
which is the point of a smoke test.
"""

from __future__ import annotations

import argparse
import json
import os

from app.agent.planners import LLMPlanner, PlannerError, build_llm_planner
from app.config import get_settings
from app.integrations.llm import is_llm_configured

DEFAULT_QUERY = "包装线PLC报警F0045怎么办"

CONFIG_VARIABLES = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    return parser


def present_variables() -> dict[str, bool]:
    """Report presence only. A value is never read into the report."""
    return {name: bool(os.environ.get(name)) for name in CONFIG_VARIABLES}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    report: dict[str, object] = {
        "query": args.query,
        "planner_mode_setting": settings.planner_mode,
        "environment_variables_present": present_variables(),
        "configured_from_settings": is_llm_configured(settings),
    }

    if not is_llm_configured(settings):
        # No endpoint is configured. Reporting a fabricated latency or plan here
        # would be worse than reporting nothing, so the gate stops.
        report["status"] = "REAL_LLM_GATE_NOT_RUN"
        report["reason"] = (
            "no LLM provider is configured; set LLM_BASE_URL, LLM_API_KEY and "
            "LLM_MODEL to run the real smoke test"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    try:
        planner: LLMPlanner = build_llm_planner(settings)
    except PlannerError as exc:
        report["status"] = "PLANNER_NOT_BUILT"
        report["error_code"] = exc.code_value
        report["error"] = exc.message
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    report["provider"] = planner.provider.describe()

    try:
        result = planner.plan(args.query)
    except PlannerError as exc:
        report["status"] = "PLAN_FAILED"
        report["error_code"] = exc.code_value
        report["error"] = exc.message
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    except Exception as exc:  # pragma: no cover - environment dependent
        report["status"] = "PLAN_FAILED"
        report["error_code"] = "UNEXPECTED_ERROR"
        report["error"] = type(exc).__name__
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    report.update(
        {
            "status": "OK",
            "planner": result.planner,
            "provider_id": result.provider,
            "model": result.model,
            "planner_latency_ms": result.latency_ms,
            "intent": result.plan.intent.value if result.plan.intent else None,
            "planned_tools": result.plan.tool_names,
            "planned_calls": [
                {
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                    "reason": call.reason,
                }
                for call in result.plan.tool_calls
            ],
        }
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
