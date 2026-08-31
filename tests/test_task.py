from __future__ import annotations

from pathlib import Path

import pytest

from benchtrace.task import TaskValidationError, load_task, load_tasks

VALID_TASK = """\
schema_version: "1.0"
id: weather
prompt_template: "Weather in {city}?"
step_budget: 3
tools:
  - name: weather
    description: Look up weather
    parameters:
      type: object
      properties:
        city: {type: string}
cases:
  - id: paris
    input: {city: Paris}
    expected:
      answer: sunny
graders:
  - type: exact_match
"""


def test_load_and_render_yaml_task(tmp_path: Path) -> None:
    task_file = tmp_path / "weather.yaml"
    task_file.write_text(VALID_TASK, encoding="utf-8")

    task = load_task(task_file)
    request = task.request_for(task.cases[0], seed=42)

    assert task.id == "weather"
    assert request.messages[-1].content == "Weather in Paris?"
    assert request.expected == {"answer": "sunny"}
    assert request.seed == 42
    assert request.tools[0].name == "weather"
    assert load_tasks(tmp_path) == [task]


def test_validation_error_names_offending_file_and_line(tmp_path: Path) -> None:
    task_file = tmp_path / "bad.yaml"
    task_file.write_text(VALID_TASK.replace("step_budget: 3", "step_budget: 0"), encoding="utf-8")

    with pytest.raises(TaskValidationError) as captured:
        load_task(task_file)

    assert f"{task_file}:4:" in str(captured.value)
    assert "greater than or equal to 1" in str(captured.value)


def test_yaml_syntax_error_names_line(tmp_path: Path) -> None:
    task_file = tmp_path / "broken.yaml"
    task_file.write_text("id: [unterminated\n", encoding="utf-8")

    with pytest.raises(TaskValidationError) as captured:
        load_task(task_file)

    assert str(task_file) in str(captured.value)
    assert captured.value.line == 2


def test_template_field_must_be_available(tmp_path: Path) -> None:
    task_file = tmp_path / "missing.yaml"
    task_file.write_text(VALID_TASK.replace("{city}", "{country}"), encoding="utf-8")

    with pytest.raises(TaskValidationError, match="country"):
        load_task(task_file)


def test_committed_tool_suite_has_thirty_cases() -> None:
    suite = load_tasks(Path(__file__).parents[1] / "tasks")

    assert len(suite) == 3
    assert sum(len(task.cases) for task in suite) == 30
    assert all(task.tools for task in suite)
    assert all(task.step_budget > 1 for task in suite)
