import gc
import json
import logging
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Optional, Any, Dict, List
from urllib.parse import urlparse
import asyncio

if TYPE_CHECKING:
	from playwright.async_api import Page, ElementHandle, Locator, Frame

from browser_use.dom.views import (
	DOMBaseNode,
	DOMElementNode,
	DOMState,
	DOMTextNode,
	ClickableElements,
	SelectorMap,
)
from browser_use.utils import time_execution_async, time_execution_sync

logger = logging.getLogger(__name__)


@dataclass
class ViewportInfo:
	width: int
	height: int


class DomService:
	def __init__(self, page: 'Page'):
		self.page = page
		self.xpath_cache = {}

		self.js_code = resources.read_text('browser_use.dom', 'buildDomTree.js')

	# region - Clickable elements
	@time_execution_async('--get_clickable_elements')
	async def get_clickable_elements(
		self,
		focus_element: int = -1,
		viewport_expansion: int = 500,
		highlight_elements: bool = True,
	) -> ClickableElements:
		"""Get all clickable elements from the page."""
		# Add a delay to ensure page is fully loaded and scripts have run
		await asyncio.sleep(0.5)
		
		# Force a DOM update to ensure elements are properly rendered
		await self.page.evaluate("""() => {
			// Force a reflow
			document.body.offsetHeight;
			// Force a repaint
			window.getComputedStyle(document.body);
		}""")
		
		# Get the DOM element tree
		element_tree = await self._get_element_tree()
		
		# Highlight elements if requested
		if highlight_elements:
			await self._add_element_highlights(element_tree)
		
		# Create selector map
		selector_map = {}
		for node in element_tree.descendants():
			if isinstance(node, DOMElementNode) and node.highlight_index is not None:
				selector_map[node.highlight_index] = node
		
		logger.info(f"Created selector map with {len(selector_map)} elements")
		
		# Ensure we have elements in the selector map
		if not selector_map and element_tree:
			logger.warning("No elements in selector map - retrying with highlight indexing")
			# Try re-indexing elements
			await self._reindex_elements(element_tree)
			
			# Update selector map
			selector_map = {}
			for node in element_tree.descendants():
				if isinstance(node, DOMElementNode) and node.highlight_index is not None:
					selector_map[node.highlight_index] = node
			
			logger.info(f"Updated selector map with {len(selector_map)} elements after reindexing")
		
		return ClickableElements(element_tree=element_tree, selector_map=selector_map)

	async def _reindex_elements(self, element_tree: DOMBaseNode) -> None:
		"""Reindex elements to ensure they have highlight indices."""
		index = 0
		for node in element_tree.descendants():
			if isinstance(node, DOMElementNode) and self._is_interactable(node):
				node.highlight_index = index
				index += 1
				
		# Try to apply highlights to the page
		try:
			elements_with_index = [node for node in element_tree.descendants() 
							  if isinstance(node, DOMElementNode) and node.highlight_index is not None]
			
			for node in elements_with_index:
				await self._highlight_element(node)
		except Exception as e:
			logger.warning(f"Error applying highlights during reindexing: {e}")

	def _is_interactable(self, node: DOMElementNode) -> bool:
		"""Check if an element is likely to be interactable."""
		if not node.tag_name:
			return False
		
		# Common interactable elements
		interactable_tags = {
			'a', 'button', 'input', 'select', 'textarea', 'option',
			'label', 'form', 'details', 'summary', 'dialog', 
		}
		
		# Elements that might be interactable based on attributes
		potentially_interactable = {
			'div', 'span', 'li', 'img', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
		}
		
		# Check if the element is directly interactable
		if node.tag_name.lower() in interactable_tags:
			return True
		
		# Check if the element has interactable attributes
		if node.tag_name.lower() in potentially_interactable:
			attrs = node.attributes
			
			# Check for click-related attributes
			if any(attr in attrs for attr in ['onclick', 'role', 'tabindex']):
				return True
			
			# Check for specific roles that indicate interactivity
			if attrs.get('role') in ['button', 'link', 'checkbox', 'radio', 'menuitem', 'tab']:
				return True
			
			# Check for specific classes that might indicate interactivity
			classes = attrs.get('class', '').lower()
			if any(cls in classes for cls in ['button', 'btn', 'link', 'clickable', 'selectable']):
				return True
		
		return False

	@time_execution_async('--get_cross_origin_iframes')
	async def get_cross_origin_iframes(self) -> list[str]:
		# invisible cross-origin iframes are used for ads and tracking, dont open those
		hidden_frame_urls = await self.page.locator('iframe').filter(visible=False).evaluate_all('e => e.map(e => e.src)')

		is_ad_url = lambda url: any(
			domain in urlparse(url).netloc for domain in ('doubleclick.net', 'adroll.com', 'googletagmanager.com')
		)

		return [
			frame.url
			for frame in self.page.frames
			if urlparse(frame.url).netloc  # exclude data:urls and about:blank
			and urlparse(frame.url).netloc != urlparse(self.page.url).netloc  # exclude same-origin iframes
			and frame.url not in hidden_frame_urls  # exclude hidden frames
			and not is_ad_url(frame.url)  # exclude most common ad network tracker frame URLs
		]

	@time_execution_async('--build_dom_tree')
	async def _build_dom_tree(
		self,
		highlight_elements: bool,
		focus_element: int,
		viewport_expansion: int,
	) -> tuple[DOMElementNode, SelectorMap]:
		if await self.page.evaluate('1+1') != 2:
			raise ValueError('The page cannot evaluate javascript code properly')

		if self.page.url == 'about:blank':
			# short-circuit if the page is a new empty tab for speed, no need to inject buildDomTree.js
			return (
				DOMElementNode(
					tag_name='body',
					xpath='',
					attributes={},
					children=[],
					is_visible=False,
					parent=None,
				),
				{},
			)

		# NOTE: We execute JS code in the browser to extract important DOM information.
		#       The returned hash map contains information about the DOM tree and the
		#       relationship between the DOM elements.
		debug_mode = logger.getEffectiveLevel() == logging.DEBUG
		args = {
			'doHighlightElements': highlight_elements,
			'focusHighlightIndex': focus_element,
			'viewportExpansion': viewport_expansion,
			'debugMode': debug_mode,
		}

		try:
			eval_page: dict = await self.page.evaluate(self.js_code, args)
		except Exception as e:
			logger.error('Error evaluating JavaScript: %s', e)
			raise

		# Only log performance metrics in debug mode
		if debug_mode and 'perfMetrics' in eval_page:
			logger.debug(
				'DOM Tree Building Performance Metrics for: %s\n%s',
				self.page.url,
				json.dumps(eval_page['perfMetrics'], indent=2),
			)

		return await self._construct_dom_tree(eval_page)

	@time_execution_async('--construct_dom_tree')
	async def _construct_dom_tree(
		self,
		eval_page: dict,
	) -> tuple[DOMElementNode, SelectorMap]:
		js_node_map = eval_page['map']
		js_root_id = eval_page['rootId']

		selector_map = {}
		node_map = {}

		for id, node_data in js_node_map.items():
			node, children_ids = self._parse_node(node_data)
			if node is None:
				continue

			node_map[id] = node

			if isinstance(node, DOMElementNode) and node.highlight_index is not None:
				selector_map[node.highlight_index] = node

			# NOTE: We know that we are building the tree bottom up
			#       and all children are already processed.
			if isinstance(node, DOMElementNode):
				for child_id in children_ids:
					if child_id not in node_map:
						continue

					child_node = node_map[child_id]

					child_node.parent = node
					node.children.append(child_node)

		html_to_dict = node_map[str(js_root_id)]

		del node_map
		del js_node_map
		del js_root_id

		gc.collect()

		if html_to_dict is None or not isinstance(html_to_dict, DOMElementNode):
			raise ValueError('Failed to parse HTML to dictionary')

		return html_to_dict, selector_map

	def _parse_node(
		self,
		node_data: dict,
	) -> tuple[Optional[DOMBaseNode], list[int]]:
		if not node_data:
			return None, []

		# Process text nodes immediately
		if node_data.get('type') == 'TEXT_NODE':
			text_node = DOMTextNode(
				text=node_data['text'],
				is_visible=node_data['isVisible'],
				parent=None,
			)
			return text_node, []

		# Process coordinates if they exist for element nodes

		viewport_info = None

		if 'viewport' in node_data:
			viewport_info = ViewportInfo(
				width=node_data['viewport']['width'],
				height=node_data['viewport']['height'],
			)

		element_node = DOMElementNode(
			tag_name=node_data['tagName'],
			xpath=node_data['xpath'],
			attributes=node_data.get('attributes', {}),
			children=[],
			is_visible=node_data.get('isVisible', False),
			is_interactive=node_data.get('isInteractive', False),
			is_top_element=node_data.get('isTopElement', False),
			is_in_viewport=node_data.get('isInViewport', False),
			highlight_index=node_data.get('highlightIndex'),
			shadow_root=node_data.get('shadowRoot', False),
			parent=None,
			viewport_info=viewport_info,
		)

		children_ids = node_data.get('children', [])

		return element_node, children_ids
