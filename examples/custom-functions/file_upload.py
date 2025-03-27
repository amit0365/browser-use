import os
import sys
from pathlib import Path

from browser_use.agent.views import ActionResult

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import logging

from langchain_openai import ChatOpenAI

from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig

logger = logging.getLogger(__name__)


# Initialize controller first
browser = Browser(
	config=BrowserConfig(
		headless=False,
		# chrome_instance_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
	),
)

controller = Controller()


@controller.action(
	'Upload file to interactive element with file path ',
)
async def upload_file(index: int, path: str, browser: BrowserContext, available_file_paths: list[str]):
	if path not in available_file_paths:
		return ActionResult(error=f'File path {path} is not available')

	if not os.path.exists(path):
		return ActionResult(error=f'File {path} does not exist')

	dom_el = await browser.get_dom_element_by_index(index)
	page = await browser.get_current_page()

	# Try file chooser approach first
	try:
		# Get the clickable element
		click_el = await browser.get_locate_element(dom_el)
		if click_el is None:
			return ActionResult(error=f'No clickable element found at index {index}')

		# Setup file chooser listener with a small timeout and click the element
		async with page.expect_file_chooser(timeout=2000) as fc_info:  # 2 second timeout
			await click_el.click()

		file_chooser = await fc_info.value
		await file_chooser.set_files(path)

		msg = f'Successfully uploaded file to index {index} using file chooser'
		logger.info(msg)
		return ActionResult(extracted_content=msg, include_in_memory=True)
	except Exception as e:
		logger.info(f'File chooser approach failed: {str(e)}')
		# Continue to try direct input approach

	# If file chooser fails, try direct file input approach
	file_upload_dom_el = dom_el.get_file_upload_element()
	if file_upload_dom_el is not None:
		file_upload_el = await browser.get_locate_element(file_upload_dom_el)
		if file_upload_el is not None:
			try:
				await file_upload_el.set_input_files(path)
				msg = f'Successfully uploaded file to index {index} using direct input'
				logger.info(msg)
				return ActionResult(extracted_content=msg, include_in_memory=True)
			except Exception as e:
				msg = f'Direct file input approach failed: {str(e)}'
				logger.info(msg)
				return ActionResult(error=msg)

	return ActionResult(error=f'Both file upload approaches failed for index {index}')


@controller.action('Read the file content of a file given a path')
async def read_file(path: str, available_file_paths: list[str]):
	if path not in available_file_paths:
		return ActionResult(error=f'File path {path} is not available')

	with open(path, 'r') as f:
		content = f.read()
	msg = f'File content: {content}'
	logger.info(msg)
	return ActionResult(extracted_content=msg, include_in_memory=True)


def create_file(file_type: str = 'txt'):
	with open(f'tmp.{file_type}', 'w') as f:
		f.write('test')
	file_path = Path.cwd() / f'tmp.{file_type}'
	logger.info(f'Created file: {file_path}')
	return str(file_path)


async def file_download_listener(path: str):
	print(f'File downloaded: {path}')


async def main():
	task = 'Go to https://v0-download-and-upload-text.vercel.app/ and first download the file and then upload that file to the upload text file field'

	# available_file_paths = ['./sample-file.txt']

	model = ChatOpenAI(model='gpt-4o')

	context = BrowserContext(
		browser=browser,
		config=BrowserContextConfig(
			save_downloads_path='./downloads/browser_use/12312312312/1312312',
		),
		register_file_download_listener_function=file_download_listener,
	)

	agent = Agent(
		task=task,
		llm=model,
		controller=controller,
		browser_context=context,
		# available_file_paths=available_file_paths,
	)

	await agent.run()

	input('Press Enter to close...')

	await browser.close()


if __name__ == '__main__':
	asyncio.run(main())
