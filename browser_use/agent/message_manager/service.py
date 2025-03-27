"""
@file service.py
@package browser_use.agent.message_manager

@brief
Manages message history, message metadata, and related tasks for the AI agent.
This includes:
  - Initialization of system, context, and example messages
  - Storing and retrieving messages with token counts
  - Adding new tasks, state messages, and model outputs
  - Maintaining conversation context for the agent

@details
## Purpose
The MessageManager class orchestrates the conversation between a user (or system),
the AI model, and any intermediary tool calls (e.g., clicking buttons). 
It ensures the right messages are injected with the correct metadata.

## Notable Changes
- Removed references to a final "AgentOutput" tool call. 
  Previously, the code injected a final tool call named "AgentOutput" into the conversation history. 
  This step eliminates that usage, storing the final output in a regular AIMessage instead.

## Implementation details
- `_init_messages`: Initializes basic messages (system prompt, context, example, placeholders).
- `add_state_message`: Adds browser state or action results.
- `add_model_output`: Now stores final data in `AIMessage(content=...)` instead of adding a "AgentOutput" function call.
- `_add_message_with_tokens`: Utility to compute approximate token usage and store metadata.

@notes
- The changes here preserve the intermediate function calls (like "click_element") but remove the forced final "AgentOutput" call, 
  in accordance with the plan to remove or rewrite the last injection of "AgentOutput".
- This file is used by other modules in the agent code to handle conversation flow.

@license
MIT License

@author
Browser Use Contributors
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel

from browser_use.agent.message_manager.views import MessageMetadata
from browser_use.agent.prompts import AgentMessagePrompt
from browser_use.agent.views import ActionResult, AgentOutput, AgentStepInfo, MessageManagerState
from browser_use.browser.views import BrowserState
from browser_use.utils import time_execution_sync

logger = logging.getLogger(__name__)


class MessageManagerSettings(BaseModel):
    """
    @class MessageManagerSettings

    @brief
    Defines configuration for MessageManager regarding token usage, included attributes, etc.

    @details
    - max_input_tokens: Maximum tokens allowed in the input.
    - estimated_characters_per_token: Rough approximation for text->token ratio.
    - image_tokens: Approx tokens used by images in the conversation.
    - include_attributes: Which HTML attributes are included in the conversation (like 'title', 'aria-label', etc.).
    - message_context: Optional extra context appended to conversation start.
    - sensitive_data: Dictionary for substituting sensitive values with placeholders.
    - available_file_paths: If the agent can upload files, these paths are allowed.
    """
    max_input_tokens: int = 128000
    estimated_characters_per_token: int = 3
    image_tokens: int = 800
    include_attributes: list[str] = []
    message_context: Optional[str] = None
    sensitive_data: Optional[Dict[str, str]] = None
    available_file_paths: Optional[List[str]] = None


class MessageManager:
    """
    @class MessageManager

    @brief
    Manages conversation messages between system, user, and AI, including
    initialization, storing outputs, adding state info, and computing tokens.

    @details
    The MessageManager:
      - Sets up initial conversation with system prompt, optional context, and placeholders.
      - Adds new tasks, state messages (like browser info or partial results), 
        model outputs, etc.
      - Enforces maximum token constraints.
      - Filters out or substitutes sensitive data.
      - Maintains message history in a @ref MessageManagerState object.
    """

    def __init__(
        self,
        task: str,
        system_message: SystemMessage,
        settings: MessageManagerSettings = MessageManagerSettings(),
        state: MessageManagerState = MessageManagerState(),
    ):
        """
        @param task: Main high-level user task or instruction
        @param system_message: The system prompt (guide for the model)
        @param settings: see @ref MessageManagerSettings
        @param state: see @ref MessageManagerState
        """
        self.task = task
        self.settings = settings
        self.state = state
        self.system_prompt = system_message

        # Only initialize messages if state is empty
        if len(self.state.history.messages) == 0:
            self._init_messages()

    def _init_messages(self) -> None:
        """
        Initializes the message history with system prompt, optional context, the main task,
        placeholders, and example usage of function calls (excluding final 'AgentOutput' references).
        """

        # Step 1: Add system prompt
        self._add_message_with_tokens(self.system_prompt)

        # Step 2: If user provided message context, add it
        if self.settings.message_context:
            context_message = HumanMessage(
                content='Context for the task' + self.settings.message_context
            )
            self._add_message_with_tokens(context_message)

        # Step 3: Add the main task as a HumanMessage
        task_message = HumanMessage(
            content=(
                f'Your ultimate task is: """{self.task}""". '
                'If you achieved your ultimate task, stop everything and use the done action in the next step to complete the task. '
                'If not, continue as usual.'
            )
        )
        self._add_message_with_tokens(task_message)

        # Step 4: If sensitive data placeholders are configured, mention them
        if self.settings.sensitive_data:
            info = (
                f'Here are placeholders for sensitve data: {list(self.settings.sensitive_data.keys())}'
            )
            info += 'To use them, write <secret>the placeholder name</secret>'
            info_message = HumanMessage(content=info)
            self._add_message_with_tokens(info_message)

        # Step 5: Example placeholders. We can show a mock function call to illustrate usage.
        placeholder_message = HumanMessage(content='Example output:')
        self._add_message_with_tokens(placeholder_message)

        # Provide an example tool call referencing something other than "AgentOutput"
        # This shows how to produce an intermediate function call like "click_element"
        example_tool_calls = [
            {
                'name': 'example_function_call',
                'args': {
                    'current_state': {
                        'evaluation_previous_goal': 'Success - I opened the first page',
                        'memory': 'Starting new task, completed 1/10 steps so far',
                        'next_goal': 'Click on a certain element',
                    },
                    'action': [{'click_element': {'index': 0}}],
                },
                'id': str(self.state.tool_id),
                'type': 'tool_call',
            }
        ]

        example_tool_call = AIMessage(
            content='',
            tool_calls=example_tool_calls,
        )
        self._add_message_with_tokens(example_tool_call)

        # Provide a tool message example
        self.add_tool_message(content='Browser started')

        # Another placeholder
        placeholder_message = HumanMessage(content='[Your task history memory starts here]')
        self._add_message_with_tokens(placeholder_message)

        # If the user configured available file paths, mention them
        if self.settings.available_file_paths:
            filepaths_msg = HumanMessage(
                content=f'Here are file paths you can use: {self.settings.available_file_paths}'
            )
            self._add_message_with_tokens(filepaths_msg)

    def add_new_task(self, new_task: str) -> None:
        """
        Adds a new high-level user task mid-way through the conversation.
        The conversation continues with the existing context, but the 
        user clarifies or overrides the final goal with the new task.
        """
        content = (
            f'Your new ultimate task is: """{new_task}""". '
            f'Take the previous context into account and finish your new ultimate task. '
        )
        msg = HumanMessage(content=content)
        self._add_message_with_tokens(msg)
        self.task = new_task

    @time_execution_sync('--add_state_message')
    def add_state_message(
        self,
        state: BrowserState,
        result: Optional[List[ActionResult]] = None,
        step_info: Optional[AgentStepInfo] = None,
        use_vision=True,
    ) -> None:
        """
        Adds the browser state as a human message, optionally with the last action result.

        @param state: Current browser state
        @param result: Optional results from the previous action(s)
        @param step_info: Additional data about the step number, if relevant
        @param use_vision: Whether vision (screenshots) is enabled
        """
        # If keep in memory was specified in the result, we add them to the conversation
        if result:
            for r in result:
                if r.include_in_memory:
                    if r.extracted_content:
                        msg = HumanMessage(content='Action result: ' + str(r.extracted_content))
                        self._add_message_with_tokens(msg)
                    if r.error:
                        if r.error.endswith('\n'):
                            r.error = r.error[:-1]
                        last_line = r.error.split('\n')[-1]
                        msg = HumanMessage(content='Action error: ' + last_line)
                        self._add_message_with_tokens(msg)
                    # reset result so it doesn't get double appended
                    result = None

        # Add the state message
        state_message = AgentMessagePrompt(
            state,
            result,
            include_attributes=self.settings.include_attributes,
            step_info=step_info,
        ).get_user_message(use_vision)
        self._add_message_with_tokens(state_message)

    def add_model_output(self, model_output: AgentOutput) -> None:
        """
        Add the model output as a final or intermediate AI message, 
        omitting the forced final 'AgentOutput' function call. 
        Instead we store the final data in the content as JSON.
        
        @param model_output: The model's parsed output object
        """
        # Convert entire model_output to JSON for the AIMessage content
        final_data = model_output.model_dump_json(exclude_unset=True)
        msg = AIMessage(
            content=final_data,
            tool_calls=[]
        )
        self._add_message_with_tokens(msg)

    def add_plan(self, plan: Optional[str], position: int | None = None) -> None:
        """
        Optionally add a 'plan' from a planner LLM. 
        This is placed as an AIMessage in the conversation for reference.

        @param plan: The text of the plan
        @param position: Where to insert in the conversation; None => end
        """
        if plan:
            msg = AIMessage(content=plan)
            self._add_message_with_tokens(msg, position)

    @time_execution_sync('--get_messages')
    def get_messages(self) -> List[BaseMessage]:
        """
        Return the current conversation as a list of BaseMessages.

        We also log token usage for debugging.
        """
        msg = [m.message for m in self.state.history.messages]
        total_input_tokens = 0
        logger.debug(f'Messages in history: {len(self.state.history.messages)}:')
        for m in self.state.history.messages:
            total_input_tokens += m.metadata.tokens
            logger.debug(f'{m.message.__class__.__name__} - Token count: {m.metadata.tokens}')
        logger.debug(f'Total input tokens: {total_input_tokens}')
        return msg

    def _add_message_with_tokens(self, message: BaseMessage, position: int | None = None) -> None:
        """
        Internal utility to add a message with computed token count to the conversation.

        @param message: The message object (SystemMessage, HumanMessage, AIMessage, or ToolMessage)
        @param position: Insert index if needed, else None => append at end
        """
        # Filter out any sensitive data if configured
        if self.settings.sensitive_data:
            message = self._filter_sensitive_data(message)

        token_count = self._count_tokens(message)
        metadata = MessageMetadata(tokens=token_count)
        self.state.history.add_message(message, metadata, position)

    @time_execution_sync('--filter_sensitive_data')
    def _filter_sensitive_data(self, message: BaseMessage) -> BaseMessage:
        """
        If configured, replaces sensitive data with <secret> placeholders in the message content.

        @param message: The original BaseMessage
        @return: Possibly modified message with placeholders
        """
        def replace_sensitive(value: str) -> str:
            if not self.settings.sensitive_data:
                return value
            for key, val in self.settings.sensitive_data.items():
                if not val:
                    continue
                value = value.replace(val, f'<secret>{key}</secret>')
            return value

        if isinstance(message.content, str):
            message.content = replace_sensitive(message.content)
        elif isinstance(message.content, list):
            for i, item in enumerate(message.content):
                if isinstance(item, dict) and 'text' in item:
                    item['text'] = replace_sensitive(item['text'])
                    message.content[i] = item
        return message

    def _count_tokens(self, message: BaseMessage) -> int:
        """
        Approximate token counting. 
        For actual, we'd use a model-specific tokenizer, but here we do a rough approach.

        @param message: The message for which we count tokens
        @return: Approx tokens
        """
        tokens = 0
        if isinstance(message.content, list):
            for item in message.content:
                if 'image_url' in item:
                    tokens += self.settings.image_tokens
                elif isinstance(item, dict) and 'text' in item:
                    tokens += self._count_text_tokens(item['text'])
        else:
            msg = message.content
            # If tool_calls exist, add them to the text for counting
            if hasattr(message, 'tool_calls') and message.tool_calls:
                msg += str(message.tool_calls)
            tokens += self._count_text_tokens(msg)
        return tokens

    def _count_text_tokens(self, text: str) -> int:
        """
        Very rough text->token approximation. 
        @param text: The input text
        @return: integer approximate token usage
        """
        return len(text) // self.settings.estimated_characters_per_token

    def cut_messages(self):
        """
        Called if token usage is too high. 
        We remove or shorten messages from the conversation to reduce usage.
        """
        diff = self.state.history.current_tokens - self.settings.max_input_tokens
        if diff <= 0:
            return None

        msg = self.state.history.messages[-1]

        # If the last message has an image, remove it to save tokens
        if isinstance(msg.message.content, list):
            text = ''
            for item in msg.message.content:
                if 'image_url' in item:
                    msg.message.content.remove(item)
                    diff -= self.settings.image_tokens
                    msg.metadata.tokens -= self.settings.image_tokens
                    self.state.history.current_tokens -= self.settings.image_tokens
                    logger.debug(
                        f'Removed image with {self.settings.image_tokens} tokens - now total: {self.state.history.current_tokens}/{self.settings.max_input_tokens}'
                    )
                elif 'text' in item and isinstance(item, dict):
                    text += item['text']
            msg.message.content = text
            self.state.history.messages[-1] = msg

        if diff <= 0:
            return None

        # If still over the limit, remove some text
        proportion_to_remove = diff / msg.metadata.tokens
        if proportion_to_remove > 0.99:
            raise ValueError(
                'Max token limit reached - history is too long. '
                'Please reduce the system prompt or the user content.'
            )
        logger.debug(
            f'Removing {proportion_to_remove * 100:.2f}% of the last message '
            f'({proportion_to_remove * msg.metadata.tokens:.2f} / {msg.metadata.tokens:.2f} tokens)'
        )

        content = msg.message.content
        characters_to_remove = int(len(content) * proportion_to_remove)
        content = content[:-characters_to_remove]

        self.state.history.remove_last_state_message()

        # Create a new message with the updated truncated content
        truncated_msg = HumanMessage(content=content)
        self._add_message_with_tokens(truncated_msg)

        last_msg = self.state.history.messages[-1]
        logger.debug(
            f'Added truncated message with {last_msg.metadata.tokens} tokens - now total tokens: {self.state.history.current_tokens}/{self.settings.max_input_tokens}.'
        )

    def _remove_last_state_message(self) -> None:
        """
        Removes the last human state message from history, if present.
        Typically used if the model call fails, or if we want to re-send the same state 
        without duplication.
        """
        self.state.history.remove_last_state_message()

    def add_tool_message(self, content: str) -> None:
        """
        Utility to insert a tool message in the conversation, 
        typically after the model calls an action.

        @param content: The tool's response
        """
        msg = ToolMessage(content=content, tool_call_id=str(self.state.tool_id))
        self.state.tool_id += 1
        self._add_message_with_tokens(msg)
