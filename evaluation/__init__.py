"""Agent capability evaluation for the Industrial Maintenance Agent.

This package is deliberately separate from ``tests/``. The pytest suite answers
"does the code do what it claims", which is a question about software
correctness. This package answers "does the agent choose the right actions for a
maintenance request", which is a question about capability and cannot be answered
by asserting on implementation details.

The two are complementary and neither replaces the other:

* a unit test can pass while the planner schedules a useless tool call, because
  the test asserted on the mechanism rather than the decision;
* an evaluation case can fail while every unit test passes, which is exactly the
  signal worth having.

Entry point::

    python -m evaluation.runner --planner rule --dataset evaluation/dataset.json
"""

__all__: list[str] = []
