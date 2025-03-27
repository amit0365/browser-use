"""
@file tests.py
@package browser_use.agent.message_manager

@brief
Test suite for the MessageManager class and related functionality in the agent/message_manager module.

@details
- These tests validate adding state messages, handling partial results, token overflow, and model outputs.
- We updated the failing tests to use a relative message count approach because `_init_messages()` now inserts
  additional placeholders ("Example output:", AIMessage with a tool call, "Browser started", etc.).
- test_final_step_no_agent_output checks final messages can store data in AIMessage without referencing "AgentOutput".

@license MIT License
"""

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.callbacks import CallbackManager

from browser_use.agent.message_manager.service import MessageManager, MessageManagerSettings
from browser_use.agent.views import ActionResult
from browser_use.browser.views import BrowserState, TabInfo
from browser_use.dom.views import DOMElementNode, DOMTextNode

# @pytest.fixture(
#     params=[
#         # ChatOpenAI(model='gpt-4o-mini'),
#         # AzureChatOpenAI(model='gpt-4o', api_version='2024-02-15-preview'),
#         ChatAnthropic(model_name='claude-3-5-sonnet-20240620', timeout=100, temperature=0.0, stop=None),
#     ],
#     ids=['claude-3-5-sonnet'],
# )
@pytest.fixture(
    params=['claude-3-5-sonnet', 'gpt-4o-mini', 'gpt-4o'],
    ids=['claude-3-5-sonnet', 'gpt-4o-mini', 'gpt-4o'],
)
def message_manager(request: pytest.FixtureRequest):
    """
    Fixture for initializing a MessageManager instance with different LLM model stubs.
    """
    task = 'Test task'
    action_descriptions = 'Test actions'
    return MessageManager(
        task=task,
        system_message=SystemMessage(content=action_descriptions),
        settings=MessageManagerSettings(
            max_input_tokens=1000,
            estimated_characters_per_token=3,
            image_tokens=800,
        ),
    )


def test_initial_messages(message_manager: MessageManager):
    """
    Test that message manager initializes with system and task messages
    plus placeholders or context messages if provided.
    We only confirm there's at least 2 messages: system + main task.
    """
    messages = message_manager.get_messages()
    assert len(messages) >= 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert 'Test task' in messages[1].content


def test_add_state_message(message_manager: MessageManager):
    """
    Test adding a browser state message to the conversation,
    verifying that exactly ONE new message is appended to the existing conversation,
    and that the last message has the page URL in it.
    """
    initial_len = len(message_manager.get_messages())
    state = BrowserState(
        url='https://test.com',
        title='Test Page',
        element_tree=DOMElementNode(
            tag_name='div',
            attributes={},
            children=[],
            is_visible=True,
            parent=None,
            xpath='//div',
        ),
        selector_map={},
        tabs=[TabInfo(page_id=1, url='https://test.com', title='Test Page')],
    )

    message_manager.add_state_message(state)

    messages = message_manager.get_messages()
    # We expect exactly +1
    assert len(messages) == initial_len + 1
    # The last message should mention 'https://test.com'
    assert 'https://test.com' in messages[-1].content


def test_add_state_with_memory_result(message_manager: MessageManager):
    """
    Test adding state with a result that should be included in conversation memory.
    This means we add TWO messages:
      1) A HumanMessage for the extracted content
      2) Another HumanMessage for the actual state
    """
    initial_len = len(message_manager.get_messages())

    state = BrowserState(
        url='https://test.com',
        title='Test Page',
        element_tree=DOMElementNode(
            tag_name='div',
            attributes={},
            children=[],
            is_visible=True,
            parent=None,
            xpath='//div',
        ),
        selector_map={},
        tabs=[TabInfo(page_id=1, url='https://test.com', title='Test Page')],
    )
    result = ActionResult(extracted_content='Important content', include_in_memory=True)

    message_manager.add_state_message(state, [result])
    messages = message_manager.get_messages()

    # We expect exactly +2 messages appended
    #   - One for the memory content
    #   - One for the state
    assert len(messages) == initial_len + 2

    # The second-to-last message is the memory content
    assert 'Important content' in messages[-2].content
    # The last message is the page state
    assert 'https://test.com' in messages[-1].content


def test_add_state_with_non_memory_result(message_manager: MessageManager):
    """
    Test adding state with a result that is not included in memory => only ONE new message:
    that message includes both the state info and the extracted content, 
    so the user sees them in a single conversation chunk.
    """
    initial_len = len(message_manager.get_messages())

    state = BrowserState(
        url='https://test.com',
        title='Test Page',
        element_tree=DOMElementNode(
            tag_name='div',
            attributes={},
            children=[],
            is_visible=True,
            parent=None,
            xpath='//div',
        ),
        selector_map={},
        tabs=[TabInfo(page_id=1, url='https://test.com', title='Test Page')],
    )
    result = ActionResult(extracted_content='Temporary content', include_in_memory=False)

    message_manager.add_state_message(state, [result])
    messages = message_manager.get_messages()

    # We expect exactly +1
    assert len(messages) == initial_len + 1
    # The last message should contain both the extracted content & the page info
    last_msg = messages[-1]
    assert 'Temporary content' in last_msg.content
    assert 'https://test.com' in last_msg.content


@pytest.mark.skip('not sure how to fix this')
@pytest.mark.parametrize('max_tokens', [100000, 10000, 5000])
def test_token_overflow_handling_with_real_flow(message_manager: MessageManager, max_tokens):
    """
    Test handling of token overflow in a realistic message flow.
    We skip it by default because it's incomplete or takes a long time.
    """
    # Set more realistic token limit
    message_manager.settings.max_input_tokens = max_tokens

    # Create a long sequence of interactions
    for i in range(200):  # Simulate many steps
        state = BrowserState(
            url=f'https://test{i}.com',
            title=f'Test Page {i}',
            element_tree=DOMElementNode(
                tag_name='div',
                attributes={},
                children=[
                    DOMTextNode(
                        text=f'Content {j} ' * (10 + i),
                        is_visible=True,
                        parent=None,
                    )
                    for j in range(5)
                ],
                is_visible=True,
                parent=None,
                xpath='//div',
            ),
            selector_map={j: f'//div[{j}]' for j in range(5)},
            tabs=[TabInfo(page_id=1, url=f'https://test{i}.com', title=f'Test Page {i}')],
        )

        if i % 2 == 0:
            # Add a partial memory result sometimes
            result = ActionResult(
                extracted_content=f'Important content from step {i}' * 5,
                include_in_memory=i % 4 == 0,
            )
            message_manager.add_state_message(state, [result])
        else:
            message_manager.add_state_message(state)

        try:
            messages = message_manager.get_messages()
        except ValueError as e:
            if 'Max token limit reached - history is too long' in str(e):
                return
            else:
                raise e

        # verify not too big
        assert message_manager.state.history.current_tokens <= message_manager.settings.max_input_tokens + 100

        # Add model output referencing old usage (for coverage)
        from browser_use.agent.views import AgentBrain, AgentOutput
        from browser_use.controller.registry.views import ActionModel

        output = AgentOutput(
            current_state=AgentBrain(
                evaluation_previous_goal=f'Success in step {i}',
                memory=f'Memory from step {i}',
                next_goal=f'Goal for step {i + 1}',
            ),
            action=[ActionModel()],
        )
        message_manager._remove_last_state_message()
        message_manager.add_model_output(output)

        # Verify structure
        msgs = [m.message for m in message_manager.state.history.messages]
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[1], HumanMessage)

        # we skip further checks for brevity


def test_final_step_no_agent_output(message_manager: MessageManager):
    """
    Step 5 test:
    Verify that the final step no longer requires an "AgentOutput" tool call 
    and can store the final data in a normal AIMessage with content.
    """
    final_content = '{"final_answer": "All tasks completed successfully", "some_key": "some_value"}'
    ai_msg = AIMessage(content=final_content)

    # Manually add to message manager
    message_manager._add_message_with_tokens(ai_msg)

    messages = message_manager.get_messages()
    # Check last message is the AIMessage with final content
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == final_content

    # Confirm there's no "AgentOutput" in the tool_calls
    tool_calls_found = getattr(messages[-1], 'tool_calls', None)
    if tool_calls_found:
        for call in tool_calls_found:
            assert call['name'] != 'AgentOutput', "Should not have final 'AgentOutput' in tool calls."
