"""
@file views.py
@package browser_use.agent

@brief
Data models and view classes that define the Agent's output schema, history,
and error handling. This file also includes the core data structures used
by the agent to store results, track state, and parse or load from saved state.

@details
- Contains Pydantic models: ActionResult, AgentBrain, AgentOutput, AgentHistory, AgentHistoryList
- Provides logic for storing the agent's step results, action results, and final outcome
- Includes the new `parse_final_json()` helper for optional final output parsing as JSON

@license
MIT License

@notes
- Step 4 Implementation: "Handle the Final Answer in Our Code" 
  We added `parse_final_json()` in `AgentHistoryList` to help parse final_result() as JSON.
- This ensures we meet the plan's requirement to parse the final content 
  from the conversation or action result if the user wants JSON data.
"""

from __future__ import annotations

import json
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from openai import RateLimitError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from browser_use.agent.message_manager.views import MessageManagerState
from browser_use.browser.views import BrowserStateHistory
from browser_use.controller.registry.views import ActionModel
from browser_use.dom.history_tree_processor.service import (
    DOMElementNode,
    DOMHistoryElement,
    HistoryTreeProcessor,
)
from browser_use.dom.views import SelectorMap

ToolCallingMethod = Literal['function_calling', 'json_mode', 'raw', 'auto']


class AgentSettings(BaseModel):
    """
    Configuration options for the Agent, controlling its behavior and environment.

    Fields:
      - use_vision: Whether the agent uses screenshots (vision-based).
      - use_vision_for_planner: Whether the planner model also uses screenshots.
      - save_conversation_path: Directory/path to save conversation logs.
      - save_conversation_path_encoding: File encoding for conversation logs.
      - max_failures: Maximum consecutive failures before halting.
      - retry_delay: Seconds to wait on rate-limit or resource exhaustion.
      - max_input_tokens: Maximum tokens for the input context.
      - validate_output: If True, we run a final validation step on the output.
      - message_context: Extra string appended to the system context.
      - generate_gif: Whether to produce a GIF summary of the agent's steps or not.
      - available_file_paths: Optional list of file paths for limited file uploads.
      - override_system_message: If set, replaces the default system prompt entirely.
      - extend_system_message: If set, appends instructions to the default system prompt.
      - include_attributes: HTML attributes to include in the conversation about web elements.
      - max_actions_per_step: Hard limit on how many actions the agent can produce in one step.
      - tool_calling_method: "auto", "function_calling", "json_mode", or "raw" controlling how the LLM calls tools.
      - page_extraction_llm: If the agent has a separate smaller or bigger model for page extraction to reduce cost.
      - planner_llm: Optional separate model for planning high-level steps.
      - planner_interval: The agent calls the planner every N steps.
    """

    use_vision: bool = True
    use_vision_for_planner: bool = False
    save_conversation_path: Optional[str] = None
    save_conversation_path_encoding: Optional[str] = 'utf-8'
    max_failures: int = 3
    retry_delay: int = 10
    max_input_tokens: int = 128000
    validate_output: bool = False
    message_context: Optional[str] = None
    generate_gif: bool | str = False
    available_file_paths: Optional[list[str]] = None
    override_system_message: Optional[str] = None
    extend_system_message: Optional[str] = None
    include_attributes: list[str] = [
        'title',
        'type',
        'name',
        'role',
        'aria-label',
        'placeholder',
        'value',
        'alt',
        'aria-expanded',
    ]
    max_actions_per_step: int = 10
    tool_calling_method: Optional[ToolCallingMethod] = 'auto'
    page_extraction_llm: Optional[BaseChatModel] = None
    planner_llm: Optional[BaseChatModel] = None
    planner_interval: int = 1


class AgentState(BaseModel):
    """
    Overall runtime state of an Agent, including ID, step count, failure tracking, 
    final results from actions, and conversation history.

    Fields:
      - agent_id: Unique identifier for the agent instance.
      - n_steps: Number of steps taken so far.
      - consecutive_failures: How many times in a row we've encountered an error.
      - last_result: The last set of ActionResults from the previous step(s).
      - history: The entire structured history (list of steps).
      - last_plan: A textual plan from a secondary 'planner' LLM, if any.
      - paused: Whether the agent is currently paused by user or system.
      - stopped: Whether the agent has been forcibly stopped and should not continue.

      - message_manager_state: Holds the conversation messages. 
        This is shared with the MessageManager in order to store AI, user, and tool calls.
    """

    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    n_steps: int = 1
    consecutive_failures: int = 0
    last_result: Optional[List['ActionResult']] = None
    history: AgentHistoryList = Field(default_factory=lambda: AgentHistoryList(history=[]))
    last_plan: Optional[str] = None
    paused: bool = False
    stopped: bool = False

    message_manager_state: MessageManagerState = Field(default_factory=MessageManagerState)


@dataclass
class AgentStepInfo:
    """
    Holds additional metadata for a single step, like step number out of max steps.
    The agent can read this to see if it's nearing the final step.
    """

    step_number: int
    max_steps: int

    def is_last_step(self) -> bool:
        """Returns True if the agent is at the final step."""
        return self.step_number >= self.max_steps - 1


class ActionResult(BaseModel):
    """
    Represents the result of executing an action at a single step.

    Fields:
      - is_done: If True, the agent's entire task is done.
      - success: Whether the action was successful if is_done is True.
      - extracted_content: Optionally store data extracted from the page.
      - error: If an error occurred, store a message or traceback.
      - include_in_memory: If True, incorporate the result into the conversation memory explicitly.
    """

    is_done: Optional[bool] = False
    success: Optional[bool] = None
    extracted_content: Optional[str] = None
    error: Optional[str] = None
    include_in_memory: bool = False  # whether to include in the conversation memory or not


class StepMetadata(BaseModel):
    """
    Encapsulates metadata for a single agent step, including timing and token usage.

    Fields:
      - step_start_time: Unix timestamp when step started
      - step_end_time: Unix timestamp when step ended
      - input_tokens: Approx tokens used for the step
      - step_number: The step number
    """

    step_start_time: float
    step_end_time: float
    input_tokens: int
    step_number: int

    @property
    def duration_seconds(self) -> float:
        """Duration of the step in seconds."""
        return self.step_end_time - self.step_start_time


class AgentBrain(BaseModel):
    """
    A snapshot of the agent's current chain-of-thought.
    Used to store short reasoning or memory from the last step,
    as well as the next immediate subgoal.

    Fields:
      - evaluation_previous_goal: A textual evaluation of how the last subgoal did
      - memory: Potential chain-of-thought or summary of what's done so far
      - next_goal: The next subgoal the agent is about to tackle
    """

    evaluation_previous_goal: str
    memory: str
    next_goal: str


class AgentOutput(BaseModel):
    """
    The primary AI output schema, specifying which actions to do next
    plus an AgentBrain for chain-of-thought.

    Fields:
      - current_state: The agent's chain-of-thought state
      - action: The list of actions (function calls) to execute next
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_state: AgentBrain
    action: list[ActionModel] = Field(
        ...,
        description='List of actions to execute',
        json_schema_extra={'min_items': 1},
    )

    @staticmethod
    def type_with_custom_actions(custom_actions: Type[ActionModel]) -> Type['AgentOutput']:
        """
        Dynamically create a subclass that uses the user-specified ActionModel 
        for the 'action' field.
        """
        model_ = create_model(
            'AgentOutput',
            __base__=AgentOutput,
            action=(
                list[custom_actions],
                Field(..., description='List of actions to execute', json_schema_extra={'min_items': 1}),
            ),
            __module__=AgentOutput.__module__,
        )
        model_.__doc__ = 'AgentOutput model with custom actions'
        return model_


class AgentHistory(BaseModel):
    """
    Represents a single step in the agent's run.

    Fields:
      - model_output: The AgentOutput describing what the LLM decided
      - result: The list of ActionResult(s) from executing that decision
      - state: The post-step browser state
      - metadata: Additional timing or token usage info for this step
    """

    model_output: AgentOutput | None
    result: list[ActionResult]
    state: BrowserStateHistory
    metadata: Optional[StepMetadata] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    @staticmethod
    def get_interacted_element(model_output: AgentOutput, selector_map: SelectorMap) -> list[DOMHistoryElement | None]:
        """
        For each action, if it references an index or xpath,
        match the corresponding DOM element in the current selector map.
        """
        elements = []
        for action in model_output.action:
            index = action.get_index()
            if index is not None and index in selector_map:
                el: DOMElementNode = selector_map[index]
                elements.append(HistoryTreeProcessor.convert_dom_element_to_history_element(el))
            else:
                elements.append(None)
        return elements

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Custom serialization to handle circular references (like self.state).
        We manually parse model_output, result, and state.
        """
        model_output_dump = None
        if self.model_output:
            action_dump = [action.model_dump(exclude_none=True) for action in self.model_output.action]
            model_output_dump = {
                'current_state': self.model_output.current_state.model_dump(),
                'action': action_dump,
            }

        return {
            'model_output': model_output_dump,
            'result': [r.model_dump(exclude_none=True) for r in self.result],
            'state': self.state.to_dict(),
            'metadata': self.metadata.model_dump() if self.metadata else None,
        }


class AgentHistoryList(BaseModel):
    """
    A complete list of agent steps. The entire story of the agent's run.

    Fields:
      - history: list of AgentHistory items

    Methods:
      - total_duration_seconds: sum of all step durations
      - final_result: the final string from the last step's extracted content
      - is_done: checks if the last step's result said is_done
      - etc.
    """

    history: list[AgentHistory]

    def total_duration_seconds(self) -> float:
        """Sum of all step durations in seconds."""
        total = 0.0
        for h in self.history:
            if h.metadata:
                total += h.metadata.duration_seconds
        return total

    def total_input_tokens(self) -> int:
        """
        Returns the sum of approximate token usage across all steps,
        as recorded in step metadata.
        """
        total = 0
        for h in self.history:
            if h.metadata:
                total += h.metadata.input_tokens
        return total

    def input_token_usage(self) -> list[int]:
        """List of token usage by step number."""
        return [h.metadata.input_tokens for h in self.history if h.metadata]

    def __str__(self) -> str:
        """
        Friendly string for debugging.
        """
        return f'AgentHistoryList(all_results={self.action_results()}, all_model_outputs={self.model_actions()})'

    def __repr__(self) -> str:
        return self.__str__()

    def save_to_file(self, filepath: str | Path) -> None:
        """
        Save the entire AgentHistoryList to a JSON file, 
        so we can replay or debug later.
        """
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            data = self.model_dump()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            raise e

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Serialize the entire history as a dictionary with minimal references.
        """
        return {
            'history': [h.model_dump(**kwargs) for h in self.history],
        }

    @classmethod
    def load_from_file(cls, filepath: str | Path, output_model: Type[AgentOutput]) -> 'AgentHistoryList':
        """
        Load an AgentHistoryList from JSON. 
        The `output_model` param is used to re-validate the agent's actions 
        with the custom ActionModel if needed.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for h in data['history']:
            if h['model_output']:
                if isinstance(h['model_output'], dict):
                    h['model_output'] = output_model.model_validate(h['model_output'])
                else:
                    h['model_output'] = None
            if 'interacted_element' not in h['state']:
                h['state']['interacted_element'] = None
        history = cls.model_validate(data)
        return history

    def last_action(self) -> None | dict:
        """
        Return the last tool call from the last step, if any.
        """
        if self.history and self.history[-1].model_output:
            return self.history[-1].model_output.action[-1].model_dump(exclude_none=True)
        return None

    def errors(self) -> list[str | None]:
        """
        Gather all error messages from each step's result 
        (or None if that step had no error).
        """
        errors = []
        for h in self.history:
            for r in h.result:
                if r.error:
                    errors.append(r.error)
        return errors

    def final_result(self) -> None | str:
        """
        Returns the final extracted_content from the last step if any.
        This is typically the final answer or summary text.
        """
        if self.history and self.history[-1].result[-1].extracted_content:
            return self.history[-1].result[-1].extracted_content
        return None

    def is_done(self) -> bool:
        """
        Check if the agent is done based on the last step's is_done field.
        """
        if self.history and len(self.history[-1].result) > 0:
            last_result = self.history[-1].result[-1]
            return last_result.is_done is True
        return False

    def is_successful(self) -> bool | None:
        """
        If is_done, return whether the agent completed successfully. None if not done yet.
        """
        if self.history and len(self.history[-1].result) > 0:
            last_result = self.history[-1].result[-1]
            if last_result.is_done is True:
                return last_result.success
        return None

    def has_errors(self) -> bool:
        """
        True if there's any step with an error.
        """
        return any(error is not None for error in self.errors())

    def urls(self) -> list[str | None]:
        """
        Return all visited URLs from the entire history.
        """
        return [h.state.url if h.state.url is not None else None for h in self.history]

    def screenshots(self) -> list[str | None]:
        """
        Return all screenshots from the entire history, if any.
        """
        return [h.state.screenshot if h.state.screenshot is not None else None for h in self.history]

    def action_names(self) -> list[str]:
        """
        Return the list of the first action name in each step. 
        E.g. "click_element", "done", "go_to_url".
        """
        action_names = []
        for action in self.model_actions():
            actions = list(action.keys())
            if actions:
                action_names.append(actions[0])
        return action_names

    def model_thoughts(self) -> list[AgentBrain]:
        """
        Return the chain-of-thought or mental state snapshots from each step, if present.
        """
        return [h.model_output.current_state for h in self.history if h.model_output]

    def model_outputs(self) -> list[AgentOutput]:
        """
        Return the entire list of AgentOutput objects from all steps.
        """
        return [h.model_output for h in self.history if h.model_output]

    def model_actions(self) -> list[dict]:
        """
        Return a flattened list of all actions from the entire run.
        Each item is a dictionary with the action name as key and its params,
        plus an "interacted_element" if relevant.
        """
        outputs = []
        for h in self.history:
            if h.model_output:
                for action, interacted_element in zip(h.model_output.action, h.state.interacted_element):
                    output = action.model_dump(exclude_none=True)
                    output['interacted_element'] = interacted_element
                    outputs.append(output)
        return outputs

    def action_results(self) -> list[ActionResult]:
        """
        Return all action results from the entire run in chronological order.
        """
        results = []
        for h in self.history:
            results.extend([r for r in h.result if r])
        return results

    def extracted_content(self) -> list[str]:
        """
        Return all extracted_content strings from each result in the run.
        """
        content = []
        for h in self.history:
            content.extend([r.extracted_content for r in h.result if r.extracted_content])
        return content

    def model_actions_filtered(self, include: list[str] | None = None) -> list[dict]:
        """
        Return only the actions matching the keys in 'include'.
        """
        if include is None:
            include = []
        outputs = self.model_actions()
        result = []
        for o in outputs:
            for i in include:
                if i == list(o.keys())[0]:
                    result.append(o)
        return result

    def number_of_steps(self) -> int:
        """Return how many steps were recorded in the history."""
        return len(self.history)

    def parse_final_json(self) -> Optional[dict]:
        """
        Attempt to parse `final_result()` as JSON. If final_result is valid JSON,
        return the loaded dict. Otherwise return None.

        Usage:
          if last_dict := history.parse_final_json():
              do_something_with(last_dict)

        Returns:
          - A Python dictionary if JSON parse succeeds
          - None if final_result is not valid JSON
        """
        final_str = self.final_result()
        if not final_str:
            return None

        try:
            return json.loads(final_str)
        except (json.JSONDecodeError, TypeError):
            return None


class AgentError:
    """
    A container of known error messages for agent issues and 
    a helper method to format them with optional trace.
    """

    VALIDATION_ERROR = 'Invalid model output format. Please follow the correct schema.'
    RATE_LIMIT_ERROR = 'Rate limit reached. Waiting before retry.'
    NO_VALID_ACTION = 'No valid action found'

    @staticmethod
    def format_error(error: Exception, include_trace: bool = False) -> str:
        """
        Format the error message for known or unknown exceptions.
        If include_trace is True, append traceback details.
        """
        message = ''
        if isinstance(error, ValidationError):
            return f'{AgentError.VALIDATION_ERROR}\nDetails: {str(error)}'
        if isinstance(error, RateLimitError):
            return AgentError.RATE_LIMIT_ERROR
        if include_trace:
            return f'{str(error)}\nStacktrace:\n{traceback.format_exc()}'
        return f'{str(error)}'

