from dotenv import load_dotenv

load_dotenv()
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents import AgentType
from langchain.tools import Tool


def main():
    print("Start...")
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = hub.pull("hwchase17/openai-functions-agent")
    tools = [PythonREPLTool()]
    agent = create_openai_functions_agent(llm, tools, prompt)

    # Create an agent executor by passing in the agent and tools
    python_agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # python_agent_executor.invoke({"input": "Generate and save in current working directory 1 QRcode that points to https://github.com/MatrixX-X, You have qrcode package installed already"})

    csv_agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        "episode-info.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    # csv_agent.run("how many columns are there in file episode_info.csv")

    grand_agent = AgentExecutor(
        agent=agent,  # Pass the agent here
        tools=[
            Tool(
                name="PythonAgent",
                func=python_agent_executor.invoke,
                description="""useful when you need to transform natural langauge and write from it python and execute the python code,
                        returning the results of the code execution,
                        DO NOT SEND PYTHON CODE TO THIS TOOL""",
                handle_parsing_errors=True,
            ),
            Tool(
                name="CSVAgent",
                func=csv_agent.invoke,
                description="""useful when you need to answer question over episode_info.csv file,
                            takes an input from the entire question and returns the answer after running pandas calculation""",
            ),
        ],
        llm=llm,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        handle_parsing_errors=True,
    )

    grand_agent.invoke(
        {
            "input": """how many rows are there in file episode_info.csv"""
        }
    )


if __name__ == "__main__":
    main()
