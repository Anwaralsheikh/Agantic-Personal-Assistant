# import sys
# import pydantic
# from pydantic import v1 as pydantic_v1

# # خدعة لإصلاح نقص pydantic_v1 في الإصدارات الجديدة
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
#         # سحب قالب ReAct الشهير
#         prompt = langchainhub("hwchase17/react")

#         # بناء الأجينت
#         agent = create_react_agent(
#             llm=self.llm, 
#             tools=self.tools, 
#             prompt=prompt
#         )
        
#         # إرجاع محرك التنفيذ
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
# from langchainhub import Client  # ✅ هذا يشتغل بعد تثبيت langchainhub
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

REACT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

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