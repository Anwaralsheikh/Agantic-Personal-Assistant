from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import  AgentExecutor ,create_tool_calling_agent
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import pandas as pd
import os

# ─── Web Search Tool ────────────────────────────────────────────
@tool
def web_search_tool(query: str) -> str:
    """Search the web for current information not found in project documents."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

# ─── CSV Analysis Tool ──────────────────────────────────────────
def create_csv_analysis_tool(files_path: str):
    @tool
    def csv_analysis_tool(query: str) -> str:
        """
        Analyze CSV or Excel files in the project.
        Use this for: statistics, averages, counts, trends, data summaries.
        Input: your question about the data.
        """
        if not files_path or not os.path.exists(files_path):
            return "No data files found."

        results = []
        for file in os.listdir(files_path):
            full_path = os.path.join(files_path, file)
            try:
                if file.endswith(".csv"):
                    df = pd.read_csv(full_path)
                    results.append(_summarize_df(df, file))

                elif file.endswith((".xlsx", ".xls")):
                    xl = pd.ExcelFile(full_path)
                    for sheet in xl.sheet_names:
                        df = xl.parse(sheet)
                        results.append(_summarize_df(df, f"{file} → {sheet}"))

            except Exception as e:
                results.append(f"Error reading {file}: {e}")

        return "\n\n".join(results) if results else "No CSV/Excel files found."

    return csv_analysis_tool


def _summarize_df(df: pd.DataFrame, name: str) -> str:
    
    return (
        f"File: {name} | "
        f"Rows: {df.shape[0]} | "
        f"Columns: {df.shape[1]} | "
        f"Column names: {list(df.columns)}"
    )


# ─── Agent ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful AI assistant with access to multiple tools.

Tool Priority:
1. project_search_tool  → ALWAYS search documents FIRST for any question
2. csv_analysis_tool    → for questions about data, numbers, statistics in CSV/Excel files  
3. web_search_tool      → ONLY if answer not found in documents or data files

Rules:
- Always answer in the same language as the question
- Be concise and accurate
- Cite which tool provided the answer"""


class ToolCallingAgent:
    def __init__(self, llm_model, tools: list, files_path: str = None):
        self.llm = llm_model
        self.tools = tools
        self.files_path = files_path

    def _build_tools(self):
        all_tools = list(self.tools)  # project_search_tool جاي من الخارج
        all_tools.append(web_search_tool)

        if self.files_path:
            all_tools.append(create_csv_analysis_tool(self.files_path))

        return all_tools

    def get_executor(self):
        all_tools = self._build_tools()

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=all_tools,
            prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=all_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15,
            max_execution_time=120,
        )