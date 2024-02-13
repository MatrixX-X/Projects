from langchain import hub
from dotenv import load_dotenv

load_dotenv()

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from tools.tools import get_profile_url


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    template = """given the name {name_of_person} I want you to find a link to their Twitter profile page, and extract from it their username.
       In Your Final answer only the person's username"""

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 Twitter profile page",
            func=get_profile_url,  # from tools.tools
            description="useful for when you need get the Twitter Page URL",
        )
    ]

    # Get the prompt to use - you can modify this!
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )

    twitter_username = result["output"]
    return twitter_username
