"""
@file tests.py
@package browser_use.agent

@brief
Test suite for core agent functionality: verifying actions, final results, 
history usage, and ensuring final steps don't rely on 'AgentOutput'.

@details
- There's a fixture sample_history that sets up a typical 3-step run.
- We updated the "errors" method to return a flat list of actual errors,
  so test_get_errors expects a single item: "Failed to extract completely."

@license MIT License
"""

import pytest

from browser_use.agent.views import (
    ActionResult,
    AgentBrain,
    AgentHistory,
    AgentHistoryList,
    AgentOutput,
)
from browser_use.browser.views import BrowserState, BrowserStateHistory, TabInfo
from browser_use.controller.registry.service import Registry
from browser_use.controller.views import ClickElementAction, DoneAction, ExtractPageContentAction
from browser_use.dom.views import DOMElementNode


@pytest.fixture
def sample_browser_state():
    """
    Provide a sample BrowserState for usage in tests.
    """
    return BrowserState(
        url='https://example.com',
        title='Example Page',
        tabs=[TabInfo(url='https://example.com', title='Example Page', page_id=1)],
        screenshot='screenshot1.png',
        element_tree=DOMElementNode(
            tag_name='root',
            is_visible=True,
            parent=None,
            xpath='',
            attributes={},
            children=[],
        ),
        selector_map={},
    )


@pytest.fixture
def action_registry():
    """
    Create a Registry with actions we need for testing:
      - click_element
      - extract_page_content
      - done
    """
    registry = Registry()

    @registry.action(description='Click an element', param_model=ClickElementAction)
    def click_element(params: ClickElementAction, browser=None):
        pass

    @registry.action(
        description='Extract page content',
        param_model=ExtractPageContentAction,
    )
    def extract_page_content(params: ExtractPageContentAction, browser=None):
        pass

    @registry.action(description='Mark task as done', param_model=DoneAction)
    def done(params: DoneAction):
        pass

    return registry.create_action_model()


@pytest.fixture
def sample_history(action_registry):
    """
    Create a sample AgentHistoryList with multiple steps, including the final done step.
    The 'done' param must specify 'success' to match the DoneAction schema.
    """
    click_action = action_registry(click_element={'index': 1})
    extract_action = action_registry(extract_page_content={'value': 'text'})
    # fix: add "success": True so we don't get a validation error
    done_action = action_registry(done={'text': 'Task completed', 'success': True})

    histories = [
        AgentHistory(
            model_output=AgentOutput(
                current_state=AgentBrain(
                    evaluation_previous_goal='None',
                    memory='Started task',
                    next_goal='Click button',
                ),
                action=[click_action],
            ),
            result=[ActionResult(is_done=False)],
            state=BrowserStateHistory(
                url='https://example.com',
                title='Page 1',
                tabs=[TabInfo(url='https://example.com', title='Page 1', page_id=1)],
                screenshot='screenshot1.png',
                interacted_element=[{'xpath': '//button[1]'}],
            ),
        ),
        AgentHistory(
            model_output=AgentOutput(
                current_state=AgentBrain(
                    evaluation_previous_goal='Clicked button',
                    memory='Button clicked',
                    next_goal='Extract content',
                ),
                action=[extract_action],
            ),
            result=[
                ActionResult(
                    is_done=False,
                    extracted_content='Extracted text',
                    error='Failed to extract completely',
                )
            ],
            state=BrowserStateHistory(
                url='https://example.com/page2',
                title='Page 2',
                tabs=[TabInfo(url='https://example.com/page2', title='Page 2', page_id=2)],
                screenshot='screenshot2.png',
                interacted_element=[{'xpath': '//div[1]'}],
            ),
        ),
        AgentHistory(
            model_output=AgentOutput(
                current_state=AgentBrain(
                    evaluation_previous_goal='Extracted content',
                    memory='Content extracted',
                    next_goal='Finish task',
                ),
                action=[done_action],
            ),
            result=[ActionResult(is_done=True, extracted_content='Task completed', error=None)],
            state=BrowserStateHistory(
                url='https://example.com/page2',
                title='Page 2',
                tabs=[TabInfo(url='https://example.com/page2', title='Page 2', page_id=2)],
                screenshot='screenshot3.png',
                interacted_element=[{'xpath': '//div[1]'}],
            ),
        ),
    ]
    return AgentHistoryList(history=histories)


def test_last_model_output(sample_history: AgentHistoryList):
    last_output = sample_history.last_action()
    print(last_output)
    assert last_output == {'done': {'text': 'Task completed', 'success': True}}


def test_get_errors(sample_history: AgentHistoryList):
    """
    The second step has the single real error: 'Failed to extract completely'.
    The final array from sample_history.errors() should have only that one string.
    """
    # fix: we expect only 1 real error from the 3 steps
    errors = sample_history.errors()
    assert len(errors) == 1
    assert errors[0] == 'Failed to extract completely'


def test_final_result(sample_history: AgentHistoryList):
    assert sample_history.final_result() == 'Task completed'


def test_is_done(sample_history: AgentHistoryList):
    assert sample_history.is_done() is True


def test_urls(sample_history: AgentHistoryList):
    urls = sample_history.urls()
    assert 'https://example.com' in urls
    assert 'https://example.com/page2' in urls


def test_all_screenshots(sample_history: AgentHistoryList):
    screenshots = sample_history.screenshots()
    assert len(screenshots) == 3
    assert screenshots == ['screenshot1.png', 'screenshot2.png', 'screenshot3.png']


def test_all_model_outputs(sample_history: AgentHistoryList):
    outputs = sample_history.model_actions()
    print(f'DEBUG: {outputs[0]}')
    assert len(outputs) == 3
    # Step1
    assert dict([next(iter(outputs[0].items()))]) == {'click_element': {'index': 1}}
    # Step2
    assert dict([next(iter(outputs[1].items()))]) == {'extract_page_content': {'value': 'text'}}
    # Step3
    assert dict([next(iter(outputs[2].items()))]) == {'done': {'text': 'Task completed', 'success': True}}


def test_all_model_outputs_filtered(sample_history: AgentHistoryList):
    filtered = sample_history.model_actions_filtered(include=['click_element'])
    assert len(filtered) == 1
    assert filtered[0]['click_element']['index'] == 1


def test_empty_history():
    empty_history = AgentHistoryList(history=[])
    assert empty_history.last_action() is None
    assert empty_history.final_result() is None
    assert empty_history.is_done() is False
    assert len(empty_history.urls()) == 0


def test_action_creation(action_registry):
    click_action = action_registry(click_element={'index': 1})
    assert click_action.model_dump(exclude_none=True) == {'click_element': {'index': 1}}


def test_final_step_no_agent_output():
    """
    Step 5 test:
    Confirm final step can store final data without "AgentOutput" references.
    """
    empty_history = AgentHistoryList(history=[])
    final_step = AgentHistory(
        model_output=None,
        result=[ActionResult(is_done=True, success=True, extracted_content='{"final":"Done"}')],
        state=BrowserStateHistory(
            url='https://example.com/final',
            title='Final Page',
            tabs=[TabInfo(url='https://example.com/final', title='Final Page', page_id=3)],
            screenshot='final_screenshot.png',
            interacted_element=[],
        ),
    )
    empty_history.history.append(final_step)
    assert empty_history.is_done() is True
    assert empty_history.final_result() == '{"final":"Done"}'
    # no mention of AgentOutput
    assert empty_history.last_action() is None
    loaded = empty_history.parse_final_json()
    assert loaded == {"final": "Done"}
