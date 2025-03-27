import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent

load_dotenv()

# Initialize the model
llm = ChatOpenAI(
	model='gpt-4o',
	temperature=0.0,
)
# the model will see x_name and x_password, but never the actual values.
sensitive_data = {'x_name': 'tinfoillabs@gmail.com', 'x_password': 'gupta@0365'}
task = 'go to gmail.com and login with x_name and x_password then summarize last 2 emails'

agent = Agent(task=task, llm=llm, sensitive_data=sensitive_data)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())
