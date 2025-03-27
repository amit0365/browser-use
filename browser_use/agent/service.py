"""
@file service.py
@package browser_use.agent

@brief
Implements the primary Agent class and logic to handle multi-step browser tasks.

@details
- The Agent orchestrates the conversation: reading the current browser state, 
  sending messages to the LLM, parsing the LLM's intended actions, and executing them.
- The conversation ends once an action sets is_done = True in its result, 
  or if the model returns no further actions (which we treat as finishing with success=False).
- Allows for error handling, consecutive failure count, partial step re-execution, 
  planner logic, and more.

@license
MIT License
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar, Union

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, ValidationError

from browser_use.agent.gif import create_history_gif
from browser_use.agent.message_manager.service import MessageManager, MessageManagerSettings
from browser_use.agent.message_manager.utils import (
    convert_input_messages,
    extract_json_from_model_output,
    save_conversation,
)
from browser_use.agent.prompts import AgentMessagePrompt, PlannerPrompt, SystemPrompt
from browser_use.agent.views import (
    ActionResult,
    AgentError,
    AgentHistory,
    AgentHistoryList,
    AgentOutput,
    AgentSettings,
    AgentState,
    AgentStepInfo,
    StepMetadata,
    ToolCallingMethod,
)
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.browser.views import BrowserState, BrowserStateHistory
from browser_use.controller.registry.views import ActionModel
from browser_use.controller.service import Controller
from browser_use.dom.history_tree_processor.service import (
    DOMHistoryElement,
    HistoryTreeProcessor,
)
from browser_use.telemetry.service import ProductTelemetry
from browser_use.telemetry.views import (
    AgentEndTelemetryEvent,
    AgentRunTelemetryEvent,
    AgentStepTelemetryEvent,
)
from browser_use.utils import time_execution_async, time_execution_sync

load_dotenv()
logger = logging.getLogger(__name__)


def log_response(response: AgentOutput) -> None:
    """
    Utility function to log the model's response, which includes:
    - evaluation_previous_goal
    - memory
    - next_goal
    - action(s)
    """
    if 'Success' in response.current_state.evaluation_previous_goal:
        emoji = '👍'
    elif 'Failed' in response.current_state.evaluation_previous_goal:
        emoji = '⚠'
    else:
        emoji = '🤷'

    logger.info(f'{emoji} Eval: {response.current_state.evaluation_previous_goal}')
    logger.info(f'🧠 Memory: {response.current_state.memory}')
    logger.info(f'🎯 Next goal: {response.current_state.next_goal}')
    for i, action in enumerate(response.action):
        logger.info(f'🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}')


Context = TypeVar('Context')


class Agent(Generic[Context]):
    """
    Main Agent class that orchestrates multi-step browser tasks using the LLM.
    It coordinates:
      - message history,
      - calling the LLM,
      - deciding which actions to run,
      - executing them,
      - tracking results & stopping conditions (like is_done).
    """

    @time_execution_sync('--init (agent)')
    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        # Optional parameters
        browser: Browser | None = None,
        browser_context: BrowserContext | None = None,
        controller: Controller[Context] = Controller(),
        # Initial agent run parameters
        sensitive_data: Optional[Dict[str, str]] = None,
        initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
        # Cloud Callbacks
        register_new_step_callback: Union[
            Callable[['BrowserState', 'AgentOutput', int], None],  # Sync callback
            Callable[['BrowserState', 'AgentOutput', int], Awaitable[None]],  # Async callback
            None
        ] = None,
        register_done_callback: Union[
            Callable[['AgentHistoryList'], Awaitable[None]], # Async Callback
            Callable[['AgentHistoryList'], None], #Sync Callback
            None
        ] = None,
        register_external_agent_status_raise_error_callback: Callable[[], Awaitable[bool]] | None = None,
        # Agent settings
        use_vision: bool = True,
        use_vision_for_planner: bool = False,
        save_conversation_path: Optional[str] = None,
        save_conversation_path_encoding: Optional[str] = 'utf-8',
        max_failures: int = 3,
        retry_delay: int = 10,
        override_system_message: Optional[str] = None,
        extend_system_message: Optional[str] = None,
        max_input_tokens: int = 128000,
        validate_output: bool = False,
        message_context: Optional[str] = None,
        generate_gif: bool | str = False,
        available_file_paths: Optional[list[str]] = None,
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
            'data-date-format',
        ],
        max_actions_per_step: int = 10,
        tool_calling_method: Optional[ToolCallingMethod] = 'auto',
        page_extraction_llm: Optional[BaseChatModel] = None,
        planner_llm: Optional[BaseChatModel] = None,
        planner_interval: int = 1,  # Run planner every N steps
        # Inject state
        injected_agent_state: Optional[AgentState] = None,
        #
        context: Context | None = None,
    ):
        """
        @param task: High-level user request or instruction
        @param llm: The main LLM used for step-by-step logic
        @param browser: Optional pre-created Browser instance
        @param browser_context: Optional pre-created BrowserContext
        @param controller: The action registry for function calling
        @param sensitive_data: Dictionary for placeholders
        @param initial_actions: Actions to run before the LLM starts
        @param register_new_step_callback: Optional callback each time the LLM produces an action set
        @param register_done_callback: Callback once the agent is done
        @param register_external_agent_status_raise_error_callback: External status check for pausing/stopping
        @param use_vision: Whether we attach screenshots for each step
        @param override_system_message: If provided, overrides the system prompt file
        @param extend_system_message: If provided, extends the system prompt file
        @param max_failures: Consec. failure limit
        @param retry_delay: Wait time on RateLimit or ResourceExhausted
        @param max_input_tokens: Max tokens in conversation
        @param validate_output: If True, we do a validation check after final step
        @param message_context: Extra context message
        @param generate_gif: If True or string, we generate a GIF of the agent's steps
        @param available_file_paths: For limiting file uploads
        @param tool_calling_method: "auto", "function_calling", "raw", or None
        @param page_extraction_llm: Optional separate LLM for large page extractions
        @param planner_llm: Optional separate LLM for high-level planning
        @param planner_interval: Run the planner every N steps
        @param injected_agent_state: If provided, reuse existing AgentState
        @param context: Generic param for user context
        """
        if page_extraction_llm is None:
            page_extraction_llm = llm

        # Core components
        self.task = task
        self.llm = llm
        self.controller = controller
        self.sensitive_data = sensitive_data

        self.settings = AgentSettings(
            use_vision=use_vision,
            use_vision_for_planner=use_vision_for_planner,
            save_conversation_path=save_conversation_path,
            save_conversation_path_encoding=save_conversation_path_encoding,
            max_failures=max_failures,
            retry_delay=retry_delay,
            override_system_message=override_system_message,
            extend_system_message=extend_system_message,
            max_input_tokens=max_input_tokens,
            validate_output=validate_output,
            message_context=message_context,
            generate_gif=generate_gif,
            available_file_paths=available_file_paths,
            include_attributes=include_attributes,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
            page_extraction_llm=page_extraction_llm,
            planner_llm=planner_llm,
            planner_interval=planner_interval,
        )

        # Initialize state
        self.state = injected_agent_state or AgentState()

        # Action setup
        self._setup_action_models()
        self._set_browser_use_version_and_source()
        self.initial_actions = self._convert_initial_actions(initial_actions) if initial_actions else None

        # Model setup
        self._set_model_names()

        # for models without tool calling, add available actions to context
        self.available_actions = self.controller.registry.get_prompt_description()

        self.tool_calling_method = self._set_tool_calling_method()
        self.settings.message_context = self._set_message_context()

        # Initialize message manager with state
        self._message_manager = MessageManager(
            task=task,
            system_message=SystemPrompt(
                action_description=self.available_actions,
                max_actions_per_step=self.settings.max_actions_per_step,
                override_system_message=override_system_message,
                extend_system_message=extend_system_message,
            ).get_system_message(),
            settings=MessageManagerSettings(
                max_input_tokens=self.settings.max_input_tokens,
                include_attributes=self.settings.include_attributes,
                message_context=self.settings.message_context,
                sensitive_data=sensitive_data,
                available_file_paths=self.settings.available_file_paths,
            ),
            state=self.state.message_manager_state,
        )

        # Browser setup
        self.injected_browser = browser is not None
        self.injected_browser_context = browser_context is not None
        if browser_context:
            self.browser = browser
            self.browser_context = browser_context
        else:
            self.browser = browser or Browser()
            self.browser_context = BrowserContext(browser=self.browser, config=self.browser.config.new_context_config)

        # Callbacks
        self.register_new_step_callback = register_new_step_callback
        self.register_done_callback = register_done_callback
        self.register_external_agent_status_raise_error_callback = register_external_agent_status_raise_error_callback

        # Context
        self.context = context

        # Telemetry
        self.telemetry = ProductTelemetry()

        if self.settings.save_conversation_path:
            logger.info(f'Saving conversation to {self.settings.save_conversation_path}')

    def _set_message_context(self) -> str | None:
        """
        If in raw mode or something else, we can inject the available actions 
        as part of the message context. 
        """
        if self.tool_calling_method == 'raw':
            if self.settings.message_context:
                self.settings.message_context += f'\n\nAvailable actions: {self.available_actions}'
            else:
                self.settings.message_context = f'Available actions: {self.available_actions}'
        return self.settings.message_context

    def _set_browser_use_version_and_source(self) -> None:
        """
        Get the version and source of the browser-use package (git or pip in a nutshell)
        for logging/telemetry.
        """
        try:
            # First check for repository-specific files
            repo_files = ['.git', 'README.md', 'docs', 'examples']
            package_root = Path(__file__).parent.parent.parent

            # If all of these files/dirs exist, it's likely from git
            if all(Path(package_root / file).exists() for file in repo_files):
                try:
                    import subprocess
                    version = subprocess.check_output(['git', 'describe', '--tags']).decode('utf-8').strip()
                except Exception:
                    version = 'unknown'
                source = 'git'
            else:
                import pkg_resources
                version = pkg_resources.get_distribution('browser-use').version
                source = 'pip'
        except Exception:
            version = 'unknown'
            source = 'unknown'

        logger.debug(f'Version: {version}, Source: {source}')
        self.version = version
        self.source = source

    def _set_model_names(self) -> None:
        self.chat_model_library = self.llm.__class__.__name__
        self.model_name = 'Unknown'
        if hasattr(self.llm, 'model_name'):
            model = self.llm.model_name  # type: ignore
            self.model_name = model if model is not None else 'Unknown'
        elif hasattr(self.llm, 'model'):
            model = self.llm.model  # type: ignore
            self.model_name = model if model is not None else 'Unknown'

        if self.settings.planner_llm:
            if hasattr(self.settings.planner_llm, 'model_name'):
                self.planner_model_name = self.settings.planner_llm.model_name  # type: ignore
            elif hasattr(self.settings.planner_llm, 'model'):
                self.planner_model_name = self.settings.planner_llm.model  # type: ignore
            else:
                self.planner_model_name = 'Unknown'
        else:
            self.planner_model_name = None

    def _setup_action_models(self) -> None:
        """
        Setup dynamic action models from controller's registry, 
        define AgentOutput type with these actions,
        plus a 'done' action type if needed.
        """
        self.ActionModel = self.controller.registry.create_action_model()
        # Create output model with the dynamic actions
        self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)

        # used to force the done action when max_steps is reached
        self.DoneActionModel = self.controller.registry.create_action_model(include_actions=['done'])
        self.DoneAgentOutput = AgentOutput.type_with_custom_actions(self.DoneActionModel)

    def _set_tool_calling_method(self) -> Optional[ToolCallingMethod]:
        """
        Decide how the model calls tools:
        - 'auto': if OpenAI or Azure, use 'function_calling'
        - if 'deepseek-r1' or 'deepseek-reasoner', use 'raw'
        - else None
        """
        tool_calling_method = self.settings.tool_calling_method
        if tool_calling_method == 'auto':
            if 'deepseek-reasoner' in self.model_name or 'deepseek-r1' in self.model_name:
                return 'raw'
            elif self.chat_model_library == 'ChatGoogleGenerativeAI':
                return None
            elif self.chat_model_library == 'ChatOpenAI':
                return 'function_calling'
            elif self.chat_model_library == 'AzureChatOpenAI':
                return 'function_calling'
            else:
                return None
        else:
            return tool_calling_method

    def add_new_task(self, new_task: str) -> None:
        """
        Add new user instructions mid-run. 
        The conversation is not reset, but we switch to a new ultimate goal.
        """
        self._message_manager.add_new_task(new_task)

    async def _raise_if_stopped_or_paused(self) -> None:
        """
        If an external or internal stop/pause is requested, 
        we raise InterruptedError to stop the current step.
        """
        if self.register_external_agent_status_raise_error_callback:
            if await self.register_external_agent_status_raise_error_callback():
                raise InterruptedError

        if self.state.stopped or self.state.paused:
            logger.debug('Agent paused after getting state')
            raise InterruptedError

    @time_execution_async('--step (agent)')
    async def step(self, step_info: Optional[AgentStepInfo] = None) -> None:
        """
        Execute one step of the agent:
          1. Get current browser state
          2. Add state message
          3. (Optionally) run planner
          4. Call LLM for next action
          5. Execute next action(s)
          6. Store results
        """
        logger.info(f'📍 Step {self.state.n_steps}')
        state = None
        model_output = None
        result: list[ActionResult] = []
        step_start_time = time.time()
        tokens = 0

        try:
            # 1. get current state
            state = await self.browser_context.get_state()

            await self._raise_if_stopped_or_paused()

            # 2. add state message
            self._message_manager.add_state_message(state, self.state.last_result, step_info, self.settings.use_vision)

            # 3. run planner if configured
            if self.settings.planner_llm and self.state.n_steps % self.settings.planner_interval == 0:
                plan = await self._run_planner()
                # add plan message
                self._message_manager.add_plan(plan, position=-1)

            # if last step is forced, add a small note
            if step_info and step_info.is_last_step():
                msg = ('Now is your last step. Please call the "done" action with success=(true|false).')
                logger.info('Last step finishing up')
                self._message_manager._add_message_with_tokens(HumanMessage(content=msg))
                self.AgentOutput = self.DoneAgentOutput

            # 4. get next action from LLM
            input_messages = self._message_manager.get_messages()
            tokens = self._message_manager.state.history.current_tokens

            try:
                model_output = await self.get_next_action(input_messages)
                self.state.n_steps += 1

                # optional callback after we get the model action
                if self.register_new_step_callback:
                    if asyncio.iscoroutinefunction(self.register_new_step_callback):
                        await self.register_new_step_callback(state, model_output, self.state.n_steps)
                    else:
                        self.register_new_step_callback(state, model_output, self.state.n_steps)

                # if saving conversation, do it now
                if self.settings.save_conversation_path:
                    target = self.settings.save_conversation_path + f'_{self.state.n_steps}.txt'
                    save_conversation(input_messages, model_output, target, self.settings.save_conversation_path_encoding)

                # remove the last state message from history to avoid duplication
                self._message_manager._remove_last_state_message()
                await self._raise_if_stopped_or_paused()

                # 4.5 if model returns no actions, we interpret that as "no more steps," end the run
                if len(model_output.action) == 0:
                    logger.info("No actions returned => concluding agent run with success=False")
                    # store final result
                    self.state.last_result = [ActionResult(
                        is_done=True,
                        success=False,
                        extracted_content="No actions => concluding. Possibly incomplete. 'done' not explicitly called."
                    )]
                    return

                # store model output
                self._message_manager.add_model_output(model_output)

            except Exception as e:
                self._message_manager._remove_last_state_message()
                raise e

            # 5. execute next action(s)
            result = await self.multi_act(model_output.action)
            self.state.last_result = result

            # if final step
            if len(result) > 0 and result[-1].is_done:
                logger.info(f'📄 Result: {result[-1].extracted_content}')

            # no error => reset consecutive failures
            self.state.consecutive_failures = 0

        except InterruptedError:
            logger.debug('Agent paused/stopped mid-step')
            self.state.last_result = [
                ActionResult(
                    error='The agent was paused/stopped - next step must re-check state or re-issue action(s)',
                    include_in_memory=True
                )
            ]
            return
        except Exception as e:
            result = await self._handle_step_error(e)
            self.state.last_result = result

        finally:
            step_end_time = time.time()
            actions = [a.model_dump(exclude_unset=True) for a in model_output.action] if model_output else []
            self.telemetry.capture(
                AgentStepTelemetryEvent(
                    agent_id=self.state.agent_id,
                    step=self.state.n_steps,
                    actions=actions,
                    consecutive_failures=self.state.consecutive_failures,
                    step_error=[r.error for r in result if r.error] if result else ['No result'],
                )
            )
            if not result:
                return

            if state:
                metadata = StepMetadata(
                    step_number=self.state.n_steps,
                    step_start_time=step_start_time,
                    step_end_time=step_end_time,
                    input_tokens=tokens,
                )
                self._make_history_item(model_output, state, result, metadata)

    @time_execution_async('--handle_step_error (agent)')
    async def _handle_step_error(self, error: Exception) -> list[ActionResult]:
        """
        Handle errors that occur during a step, e.g. model call or action execution.
        - If it's a known error like ValueError or RateLimitError, handle or retry
        - If it’s unknown, log and increment consecutive_failures
        - Return ActionResult with error
        """
        include_trace = logger.isEnabledFor(logging.DEBUG)
        error_msg = AgentError.format_error(error, include_trace=include_trace)
        prefix = f'❌ Result failed {self.state.consecutive_failures + 1}/{self.settings.max_failures} times:\n '

        from google.api_core.exceptions import ResourceExhausted
        from openai import RateLimitError

        if isinstance(error, (ValidationError, ValueError)):
            logger.error(f'{prefix}{error_msg}')
            if 'Max token limit reached' in error_msg:
                # cut tokens from history
                self._message_manager.settings.max_input_tokens = self.settings.max_input_tokens - 500
                logger.info(
                    f'Cutting tokens from history - new max input tokens: {self._message_manager.settings.max_input_tokens}'
                )
                self._message_manager.cut_messages()
            elif 'Could not parse response' in error_msg:
                # give model a hint how output should look like
                error_msg += '\n\nReturn a valid JSON object with the required fields.'
            self.state.consecutive_failures += 1
        elif isinstance(error, (RateLimitError, ResourceExhausted)):
            logger.warning(f'{prefix}{error_msg}')
            await asyncio.sleep(self.settings.retry_delay)
            self.state.consecutive_failures += 1
        else:
            logger.error(f'{prefix}{error_msg}')
            self.state.consecutive_failures += 1

        return [ActionResult(error=error_msg, include_in_memory=True)]

    def _make_history_item(
        self,
        model_output: AgentOutput | None,
        state: BrowserState,
        result: list[ActionResult],
        metadata: Optional[StepMetadata] = None,
    ) -> None:
        """
        Create & store an AgentHistory item for this step, referencing:
          - model output
          - result
          - final browser state
        """
        if model_output:
            interacted_elements = AgentHistory.get_interacted_element(model_output, state.selector_map)
        else:
            interacted_elements = [None]

        state_history = BrowserStateHistory(
            url=state.url,
            title=state.title,
            tabs=state.tabs,
            interacted_element=interacted_elements,
            screenshot=state.screenshot,
        )

        history_item = AgentHistory(model_output=model_output, result=result, state=state_history, metadata=metadata)
        self.state.history.history.append(history_item)

    def _remove_think_tags(self, text: str) -> str:
        """
        Attempt to remove <think> ... </think> tags if present in the model's raw output.
        """
        import re
        THINK_TAGS = re.compile(r'<think>.*?</think>', re.DOTALL)
        STRAY_CLOSE_TAG = re.compile(r'.*?</think>', re.DOTALL)
        # Remove well-formed <think>...</think>
        text = re.sub(THINK_TAGS, '', text)
        # If there's an unmatched closing tag </think>, remove everything up to and including that
        text = re.sub(STRAY_CLOSE_TAG, '', text)
        return text.strip()

    def _convert_input_messages(self, input_messages: list[BaseMessage]) -> list[BaseMessage]:
        """
        Convert the message list for the relevant model, e.g. 
        if using DeepSeek reasoner, we merge messages. Otherwise, we pass them as is.
        """
        if 'deepseek-reasoner' in self.model_name or 'deepseek-r1' in self.model_name:
            return convert_input_messages(input_messages, self.model_name)
        else:
            return input_messages

    @time_execution_async('--get_next_action (agent)')
    async def get_next_action(self, input_messages: list[BaseMessage]) -> AgentOutput:
        """
        Query the LLM to get next action from the conversation so far.
        Return the parsed action structure as an AgentOutput object.

        Implementation detail: 
         - if tool_calling_method = 'raw', parse JSON from the text
         - otherwise use structured output approach
        """
        input_messages = self._convert_input_messages(input_messages)

        if self.tool_calling_method == 'raw':
            # direct text => parse JSON from the content
            output = self.llm.invoke(input_messages)
            output.content = self._remove_think_tags(str(output.content))
            try:
                parsed_json = extract_json_from_model_output(output.content)
                parsed = self.AgentOutput(**parsed_json)
            except (ValueError, ValidationError) as e:
                logger.warning(f'Failed to parse model output: {output} {str(e)}')
                raise ValueError('Could not parse response.')
        elif self.tool_calling_method is None:
            # no function calling, but structured output if we can
            structured_llm = self.llm.with_structured_output(self.AgentOutput, include_raw=True)
            response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
            parsed: AgentOutput | None = response['parsed']
        else:
            # function_calling or json_mode
            structured_llm = self.llm.with_structured_output(self.AgentOutput, include_raw=True, method=self.tool_calling_method)
            response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore

            if response.get('parsing_error') and 'raw' in response:
                raw_msg = response['raw']
                if hasattr(raw_msg, 'tool_calls') and raw_msg.tool_calls:
                    tool_call = raw_msg.tool_calls[0]
                    tool_call_name = tool_call['name']
                    tool_call_args = tool_call['args']
                    current_state = {
                        'evaluation_previous_goal': 'Executing action',
                        'memory': 'Using tool call fallback',
                        'next_goal': f'Execute {tool_call_name}',
                    }
                    action = {tool_call_name: tool_call_args}
                    parsed = self.AgentOutput(
                        current_state=current_state,
                        action=[self.ActionModel(**action)]
                    )
                else:
                    parsed = None
            else:
                parsed = response['parsed']

            if not parsed:
                try:
                    raw_content = response["raw"].content
                    parsed_json = extract_json_from_model_output(raw_content)
                    parsed = self.AgentOutput(**parsed_json)
                except Exception as e:
                    logger.warning(f'Failed to parse model output: {response["raw"].content} {str(e)}')
                    raise ValueError('Could not parse response.')

        # cut to max_actions_per_step if needed
        if len(parsed.action) > self.settings.max_actions_per_step:
            parsed.action = parsed.action[: self.settings.max_actions_per_step]

        log_response(parsed)
        return parsed

    def _log_agent_run(self) -> None:
        """
        Log the agent run start with telemetry, capturing model info,
        use_vision, etc.
        """
        logger.info(f'🚀 Starting task: {self.task}')
        logger.debug(f'Version: {self.version}, Source: {self.source}')
        self.telemetry.capture(
            AgentRunTelemetryEvent(
                agent_id=self.state.agent_id,
                use_vision=self.settings.use_vision,
                task=self.task,
                model_name=self.model_name,
                chat_model_library=self.chat_model_library,
                version=self.version,
                source=self.source,
            )
        )

    async def take_step(self) -> tuple[bool, bool]:
        """
        Take a single step in the agent loop:
        returns (is_done, is_valid)
          - is_done: bool indicating if the agent declared done
          - is_valid: if validated output passes or not
        """
        await self.step()

        if self.state.history.is_done():
            if self.settings.validate_output:
                if not await self._validate_output():
                    return True, False
            await self.log_completion()
            if self.register_done_callback:
                if asyncio.iscoroutinefunction(self.register_done_callback):
                    await self.register_done_callback(self.state.history)
                else:
                    self.register_done_callback(self.state.history)
            return True, True

        return False, False

    @time_execution_async('--run (agent)')
    async def run(self, max_steps: int = 100) -> AgentHistoryList:
        """
        Run the agent for up to max_steps. 
        - If at any point is_done is triggered, we break early.
        - If consecutive_failures >= max_failures, we stop as well.
        """
        try:
            self._log_agent_run()

            # Execute initial actions if provided
            if self.initial_actions:
                result = await self.multi_act(self.initial_actions, check_for_new_elements=False)
                self.state.last_result = result

            for step in range(max_steps):
                # check consecutive failures
                if self.state.consecutive_failures >= self.settings.max_failures:
                    logger.error(f'❌ Stopping due to {self.settings.max_failures} consecutive failures')
                    break

                # check paused/stopped
                if self.state.stopped:
                    logger.info('Agent stopped')
                    break

                while self.state.paused:
                    await asyncio.sleep(0.2)
                    if self.state.stopped:
                        break

                step_info = AgentStepInfo(step_number=step, max_steps=max_steps)
                await self.step(step_info)

                # check is_done
                if self.state.history.is_done():
                    # optional validation
                    if self.settings.validate_output and step < max_steps - 1:
                        if not await self._validate_output():
                            continue
                    await self.log_completion()
                    break
            else:
                logger.info('❌ Failed to complete task in maximum steps')

            return self.state.history
        finally:
            self.telemetry.capture(
                AgentEndTelemetryEvent(
                    agent_id=self.state.agent_id,
                    is_done=self.state.history.is_done(),
                    success=self.state.history.is_successful(),
                    steps=self.state.n_steps,
                    max_steps_reached=self.state.n_steps >= max_steps,
                    errors=self.state.history.errors(),
                    total_input_tokens=self.state.history.total_input_tokens(),
                    total_duration_seconds=self.state.history.total_duration_seconds(),
                )
            )

            await self.close()

            if self.settings.generate_gif:
                output_path: str = 'agent_history.gif'
                if isinstance(self.settings.generate_gif, str):
                    output_path = self.settings.generate_gif
                create_history_gif(task=self.task, history=self.state.history, output_path=output_path)

    @time_execution_async('--multi-act (agent)')
    async def multi_act(
        self,
        actions: list[ActionModel],
        check_for_new_elements: bool = True,
    ) -> list[ActionResult]:
        """
        Execute multiple actions from the LLM in sequence. 
        If an action triggers new elements or a new tab, we might re-check the environment.

        @param actions: The list of parsed ActionModel objects
        @param check_for_new_elements: If True, we stop if we see new elements appear mid-step
        """
        results = []

        cached_selector_map = await self.browser_context.get_selector_map()
        cached_path_hashes = set(e.hash.branch_path_hash for e in cached_selector_map.values())

        await self.browser_context.remove_highlights()

        for i, action in enumerate(actions):
            if action.get_index() is not None and i != 0:
                new_state = await self.browser_context.get_state()
                new_path_hashes = set(e.hash.branch_path_hash for e in new_state.selector_map.values())
                if check_for_new_elements and not new_path_hashes.issubset(cached_path_hashes):
                    msg = f'New elements appeared after action {i} / {len(actions)}'
                    logger.info(msg)
                    results.append(ActionResult(extracted_content=msg, include_in_memory=True))
                    break

            await self._raise_if_stopped_or_paused()

            result = await self.controller.act(
                action,
                self.browser_context,
                self.settings.page_extraction_llm,
                self.sensitive_data,
                self.settings.available_file_paths,
                context=self.context,
            )
            results.append(result)

            logger.debug(f'Executed action {i + 1} / {len(actions)}')
            if results[-1].is_done or results[-1].error or i == len(actions) - 1:
                break

            await asyncio.sleep(self.browser_context.config.wait_between_actions)

        return results

    async def _validate_output(self) -> bool:
        """
        If configured, run a quick validation with the main LLM.
        We pass the final result to the LLM and ask if it's correct or not, 
        returning True/False.
        """
        system_msg = (
            f'You are a validator of an agent who interacts with a browser. '
            f'Validate if the output of last action is what the user wanted and if the task is completed. '
            f'If something is missing or the result is incomplete, let it fail. '
            f'Return a JSON object with 2 keys: is_valid (bool) and reason (string).'
        )

        if self.browser_context.session:
            state = await self.browser_context.get_state()
            content = AgentMessagePrompt(
                state=state,
                result=self.state.last_result,
                include_attributes=self.settings.include_attributes,
            )
            msg = [SystemMessage(content=system_msg), content.get_user_message(self.settings.use_vision)]
        else:
            # if no browser session, skip
            return True

        class ValidationResult(BaseModel):
            is_valid: bool
            reason: str

        validator = self.llm.with_structured_output(ValidationResult, include_raw=True)
        response: dict[str, Any] = await validator.ainvoke(msg)  # type: ignore
        parsed: ValidationResult = response['parsed']
        is_valid = parsed.is_valid
        if not is_valid:
            logger.info(f'❌ Validator decision: {parsed.reason}')
            msg = f'The output is not correct. Reason: {parsed.reason}'
            self.state.last_result = [ActionResult(extracted_content=msg, include_in_memory=True)]
        else:
            logger.info(f'✅ Validator decision: {parsed.reason}')
        return is_valid

    async def log_completion(self) -> None:
        """
        Called once the agent signals done. 
        We finalize logs, and if self.register_done_callback is set, we call it.
        """
        logger.info('✅ Task completed')
        if self.state.history.is_successful():
            logger.info('✅ Successfully')
        else:
            logger.info('❌ Unfinished')

        if self.register_done_callback:
            if asyncio.iscoroutinefunction(self.register_done_callback):
                await self.register_done_callback(self.state.history)
            else:
                self.register_done_callback(self.state.history)

    async def rerun_history(
        self,
        history: AgentHistoryList,
        max_retries: int = 3,
        skip_failures: bool = True,
        delay_between_actions: float = 2.0,
    ) -> list[ActionResult]:
        """
        Re-run a saved history of actions with optional error handling and retry logic.
        This is more advanced usage for debugging or replays.
        """
        if self.initial_actions:
            result = await self.multi_act(self.initial_actions)
            self.state.last_result = result

        results = []

        for i, history_item in enumerate(history.history):
            goal = history_item.model_output.current_state.next_goal if history_item.model_output else ''
            logger.info(f'Replaying step {i + 1}/{len(history.history)}: goal: {goal}')

            if (
                not history_item.model_output
                or not history_item.model_output.action
                or history_item.model_output.action == [None]
            ):
                logger.warning(f'Step {i + 1}: No action to replay, skipping')
                results.append(ActionResult(error='No action to replay'))
                continue

            retry_count = 0
            while retry_count < max_retries:
                try:
                    result = await self._execute_history_step(history_item, delay_between_actions)
                    results.extend(result)
                    break

                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        error_msg = f'Step {i + 1} failed after {max_retries} attempts: {str(e)}'
                        logger.error(error_msg)
                        if not skip_failures:
                            results.append(ActionResult(error=error_msg))
                            raise RuntimeError(error_msg)
                    else:
                        logger.warning(f'Step {i + 1} failed (attempt {retry_count}/{max_retries}), retrying...')
                        await asyncio.sleep(delay_between_actions)

        return results

    async def _execute_history_step(self, history_item: AgentHistory, delay: float) -> list[ActionResult]:
        """
        Actually re-execute a single step from the saved history,
        matching elements by comparing DOM structure, etc.
        """
        state = await self.browser_context.get_state()
        if not state or not history_item.model_output:
            raise ValueError('Invalid state or model output for replay')

        updated_actions = []
        for i, action in enumerate(history_item.model_output.action):
            updated_action = await self._update_action_indices(
                history_item.state.interacted_element[i],
                action,
                state,
            )
            updated_actions.append(updated_action)
            if updated_action is None:
                raise ValueError(f'Could not find matching element {i} in current page')

        result = await self.multi_act(updated_actions)

        await asyncio.sleep(delay)
        return result

    async def _update_action_indices(
        self,
        historical_element: Optional[DOMHistoryElement],
        action: ActionModel,
        current_state: BrowserState,
    ) -> Optional[ActionModel]:
        """
        Attempt to match the historical element to the new DOM. 
        If found, update the action's index to the new location (highlight_index).
        """
        if not historical_element or not current_state.element_tree:
            return action

        current_element = HistoryTreeProcessor.find_history_element_in_tree(historical_element, current_state.element_tree)

        if not current_element or current_element.highlight_index is None:
            return None

        old_index = action.get_index()
        if old_index != current_element.highlight_index:
            action.set_index(current_element.highlight_index)
            logger.info(f'Element moved in DOM, updated index from {old_index} to {current_element.highlight_index}')

        return action

    async def load_and_rerun(self, history_file: Optional[str | Path] = None, **kwargs) -> list[ActionResult]:
        """
        Load a previously saved AgentHistoryList and rerun it with the current agent's environment.
        """
        if not history_file:
            history_file = 'AgentHistory.json'
        history = AgentHistoryList.load_from_file(history_file, self.AgentOutput)
        return await self.rerun_history(history, **kwargs)

    def save_history(self, file_path: Optional[str | Path] = None) -> None:
        """
        Save the entire agent history to a JSON file for future replay or debugging.
        """
        if not file_path:
            file_path = 'AgentHistory.json'
        self.state.history.save_to_file(file_path)

    def pause(self) -> None:
        """
        Pause the agent before the next step.
        """
        logger.info('🔄 pausing Agent ')
        self.state.paused = True

    def resume(self) -> None:
        """
        Resume the agent.
        """
        logger.info('▶️ Agent resuming')
        self.state.paused = False

    def stop(self) -> None:
        """
        Stop the agent. This sets the agent to a 'stopped' state, 
        preventing further steps.
        """
        logger.info('⏹️ Agent stopping')
        self.state.stopped = True

    def _convert_initial_actions(self, actions: List[Dict[str, Dict[str, Any]]]) -> List[ActionModel]:
        """
        Convert user-provided initial_actions from dict form to typed ActionModel.
        """
        converted_actions = []
        action_model = self.ActionModel
        for action_dict in actions:
            action_name = next(iter(action_dict))
            params = action_dict[action_name]

            # param model
            action_info = self.controller.registry.registry.actions[action_name]
            param_model = action_info.param_model

            validated_params = param_model(**params)
            action_model = self.ActionModel(**{action_name: validated_params})
            converted_actions.append(action_model)

        return converted_actions

    async def _run_planner(self) -> Optional[str]:
        """
        If planner_llm is set, produce a "plan" message analyzing the state
        and next steps. 
        This is optional. The "planner" is an additional LLM that attempts 
        to help break down tasks.
        """
        if not self.settings.planner_llm:
            return None

        planner_messages = [
            PlannerPrompt(self.controller.registry.get_prompt_description()).get_system_message(),
            *self._message_manager.get_messages()[1:],  # skip the first system
        ]

        if not self.settings.use_vision_for_planner and self.settings.use_vision:
            last_state_message: HumanMessage = planner_messages[-1]
            new_msg = ''
            if isinstance(last_state_message.content, list):
                for msg in last_state_message.content:
                    if msg['type'] == 'text': 
                        new_msg += msg['text']
                    elif msg['type'] == 'image_url':
                        continue
            else:
                new_msg = last_state_message.content
            planner_messages[-1] = HumanMessage(content=new_msg)

        from browser_use.agent.message_manager.utils import convert_input_messages
        planner_messages = convert_input_messages(planner_messages, self.planner_model_name)

        response = await self.settings.planner_llm.ainvoke(planner_messages)
        plan = str(response.content)
        if self.planner_model_name and ('deepseek-r1' in self.planner_model_name or 'deepseek-reasoner' in self.planner_model_name):
            plan = self._remove_think_tags(plan)

        try:
            import json
            plan_json = json.loads(plan)
            logger.info(f'Planning Analysis:\n{json.dumps(plan_json, indent=4)}')
        except json.JSONDecodeError:
            logger.info(f'Planning Analysis:\n{plan}')
        except Exception as e:
            logger.debug(f'Error parsing planning analysis: {e}')
            logger.info(f'Plan: {plan}')

        return plan

    @property
    def message_manager(self) -> MessageManager:
        return self._message_manager

    async def close(self):
        """
        Cleanup resources: close browser if not injected, 
        do final GC.
        """
        try:
            if self.browser_context and not self.injected_browser_context:
                await self.browser_context.close()
            if self.browser and not self.injected_browser:
                await self.browser.close()
            gc.collect()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

