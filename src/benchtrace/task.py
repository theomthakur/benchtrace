"""Validated YAML task definitions rendered into canonical trace requests."""

from __future__ import annotations

import string
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator
from yaml.error import MarkedYAMLError
from yaml.nodes import MappingNode, Node, SequenceNode

from benchtrace.trace import Message, ToolDefinition, TraceRequest


class TaskModel(BaseModel):
    """Strict immutable base for task data."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExactMatchSpec(TaskModel):
    type: Literal["exact_match"]
    expected: str = "answer"
    case_sensitive: bool = True
    strip: bool = True


class RegexSpec(TaskModel):
    type: Literal["regex"]
    expected: str = "pattern"
    flags: list[Literal["IGNORECASE", "MULTILINE", "DOTALL"]] = Field(default_factory=list)


class NumericSpec(TaskModel):
    type: Literal["numeric"]
    expected: str = "answer"
    absolute_tolerance: float = Field(default=0.0, ge=0.0)
    relative_tolerance: float = Field(default=0.0, ge=0.0)


class JsonSchemaSpec(TaskModel):
    type: Literal["json_schema"]
    expected: str = "schema"


class ToolCallSpec(TaskModel):
    type: Literal["tool_call"]
    expected: str = "tool_calls"
    allow_extra: bool = False


class LLMJudgeSpec(TaskModel):
    type: Literal["llm_judge"]
    rubric: str = Field(min_length=1)
    model: str = Field(min_length=1)
    force_same_model: bool = False


GraderSpec = Annotated[
    ExactMatchSpec | RegexSpec | NumericSpec | JsonSchemaSpec | ToolCallSpec | LLMJudgeSpec,
    Field(discriminator="type"),
]


class TaskCase(TaskModel):
    """One parameterized benchmark case."""

    id: str = Field(min_length=1)
    input: dict[str, JsonValue] = Field(default_factory=dict)
    expected: dict[str, JsonValue]
    relevant_context_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TaskDefinition(TaskModel):
    """A complete data-defined benchmark task."""

    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(min_length=1)
    description: str = ""
    system_prompt: str | None = None
    prompt_template: str = Field(min_length=1)
    cases: list[TaskCase] = Field(min_length=1)
    graders: list[GraderSpec] = Field(min_length=1)
    tools: list[ToolDefinition] = Field(default_factory=list)
    step_budget: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def unique_ids_and_valid_template(self) -> TaskDefinition:
        """Reject duplicate cases and malformed format strings at load time."""

        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique within a task")
        try:
            fields = {
                field_name
                for _, field_name, _, _ in string.Formatter().parse(self.prompt_template)
                if field_name
            }
        except ValueError as error:
            raise ValueError(f"invalid prompt_template: {error}") from error
        available = set().union(*(case.input.keys() for case in self.cases))
        missing = fields - available
        if missing:
            raise ValueError(f"prompt_template fields absent from every case: {sorted(missing)}")
        return self

    def request_for(self, case: TaskCase, *, seed: int | None = None) -> TraceRequest:
        """Render a case directly into the trace request contract."""

        try:
            prompt = self.prompt_template.format_map(case.input)
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"task {self.id!r} case {case.id!r}: cannot render prompt: {error}"
            ) from error
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.append(Message(role="user", content=prompt))
        return TraceRequest(
            messages=messages,
            tools=self.tools,
            case_input=case.input,
            expected=case.expected,
            relevant_context_ids=case.relevant_context_ids,
            seed=seed,
        )


class TaskValidationError(ValueError):
    """Task parsing error that always identifies a source path and line."""

    def __init__(self, path: Path, line: int, detail: str) -> None:
        super().__init__(f"{path}:{line}: {detail}")
        self.path = path
        self.line = line


def _line_for_location(node: Node, location: tuple[str | int, ...]) -> int:
    current = node
    for component in location:
        if isinstance(component, str) and isinstance(current, MappingNode):
            match: Node | None = None
            for key_node, value_node in current.value:
                if key_node.value == component:
                    match = value_node
                    break
            if match is None:
                return current.start_mark.line + 1
            current = match
        elif isinstance(component, int) and isinstance(current, SequenceNode):
            if component >= len(current.value):
                return current.start_mark.line + 1
            current = current.value[component]
        else:
            return current.start_mark.line + 1
    return current.start_mark.line + 1


def load_task(path: str | Path) -> TaskDefinition:
    """Load one YAML task with schema errors localized to file and line."""

    source = Path(path)
    try:
        content = source.read_text(encoding="utf-8")
    except OSError as error:
        raise TaskValidationError(source, 0, str(error)) from error
    try:
        node = yaml.compose(content)
        raw: object = yaml.safe_load(content)
    except MarkedYAMLError as error:
        line = error.problem_mark.line + 1 if error.problem_mark else 1
        raise TaskValidationError(source, line, str(error)) from error
    except yaml.YAMLError as error:
        raise TaskValidationError(source, 1, str(error)) from error
    if node is None:
        raise TaskValidationError(source, 1, "task file is empty")
    try:
        return TaskDefinition.model_validate(raw)
    except ValidationError as error:
        first = error.errors()[0]
        location = tuple(item for item in first["loc"] if isinstance(item, str | int))
        line = _line_for_location(node, location)
        raise TaskValidationError(source, line, str(error)) from error


def load_tasks(path: str | Path) -> list[TaskDefinition]:
    """Load a task file or all YAML files recursively from a directory."""

    source = Path(path)
    if source.is_file():
        return [load_task(source)]
    if not source.is_dir():
        raise TaskValidationError(source, 0, "task path does not exist")
    task_files = sorted((*source.rglob("*.yaml"), *source.rglob("*.yml")))
    if not task_files:
        raise TaskValidationError(source, 0, "no YAML task files found")
    tasks = [load_task(task_file) for task_file in task_files]
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise TaskValidationError(source, 0, "task ids must be unique across the suite")
    return tasks
