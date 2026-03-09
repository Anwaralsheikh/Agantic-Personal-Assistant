# import sys
# import pydantic
# from pydantic import v1 as pydantic_v1
# import langchain_core
# if not hasattr(langchain_core, "pydantic_v1"):
#     langchain_core.pydantic_v1 = pydantic_v1
#     sys.modules["langchain_core.pydantic_v1"] = pydantic_v1
# # from langchain.agents.agent import AgentExecutor
# # from langchain.agents import create_react_agent ,AgentExecutor
# from langchain_classic.agents import create_react_agent, AgentExecutor
# import langchainhub
# from langchain_core.tools import Tool

# class RagAgent:
#     def __init__(self, llm_model, tools: list):
#         self.llm = llm_model
#         self.tools = tools

#     def get_executor(self):
#     
#         prompt = langchainhub("hwchase17/react")

#        
#         agent = create_react_agent(
#             llm=self.llm, 
#             tools=self.tools, 
#             prompt=prompt
#         )
        
#         
#         return AgentExecutor(
#             agent=agent, 
#             tools=self.tools, 
#             verbose=True,
#             handle_parsing_errors=True
#         )

import sys
import pydantic
from pydantic import v1 as pydantic_v1

import langchain_core
if not hasattr(langchain_core, "pydantic_v1"):
    langchain_core.pydantic_v1 = pydantic_v1
    sys.modules["langchain_core.pydantic_v1"] = pydantic_v1

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

REACT_TEMPLATE = """You are a helpful assistant. Answer questions using ONLY the provided tools.

You have access to the following tools:
{tools}

STRICT Rules:
- Use ONLY this format, no exceptions:

Thought: [your reasoning]
Action: [tool name from: {tool_names}]
Action Input: [input to the tool]
Observation: [tool result]
... (repeat Thought/Action/Action Input/Observation if needed)
Thought: I now know the final answer
Final Answer: [your answer here]

- After receiving an Observation, if you have enough info → write "Final Answer:" IMMEDIATELY
- NEVER write "Action: None"
- NEVER repeat the question
- ALWAYS end with "Final Answer:"

Begin!

Question: {input}
Thought: I should search the documents first.
{agent_scratchpad}"""

class RagAgent:
    def __init__(self, llm_model, tools: list):
        self.llm = llm_model
        self.tools = tools

    def get_executor(self):
        prompt = PromptTemplate.from_template(REACT_TEMPLATE)

        agent = create_react_agent(
            llm=self.llm, 
            tools=self.tools, 
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            max_execution_time=60,
        )