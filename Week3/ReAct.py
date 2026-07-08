from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain.tools import tool
from dotenv import load_dotenv
import requests
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
)

search = DuckDuckGoSearchRun()

@tool
def calc_tool(expr: str) -> str:
    """Use only and only for arithmetic expressions like 2+2 or 25*(3+2)."""
    return str(eval(expr)) 

@tool
def search_tool(query: str) -> str:
    """Use for current events, weather, recent facts, news."""
    return search.run(query)

@tool
def wiki_tool(query: str) -> str: 
    """
    Use for exact entity names only.
    Examples:
    Napoleon
    Adolf Hitler
    World War II
    """
    try:
        headers = {
            "User-Agent": "ReAct/1.0"
        }

        search_url = "https://en.wikipedia.org/w/api.php"

        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": query,
            "format": "json"
        }

        r = requests.get(search_url, params=params, headers=headers, timeout=10)
        r.raise_for_status()

        data = r.json()
        pages = data["query"]["pages"]

        page = next(iter(pages.values()))

        if "missing" in page:
            return f"No Wikipedia page found for: {query}"

        text = page.get("extract", "").strip()

        if not text:
            return "No summary available."

        return text[:1000]
    
    except Exception as e:
        return "Wikipedia API failed."

tools = [search_tool, wiki_tool, calc_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True 
)

print("Agent: Hello! Ask me something (Enter 'q' to quit).")
while True:
    query = input("\nUser: ").strip()

    if query == 'q':
        print("Agent: Goodbye!")
        break

    result = agent.run(query)
    print("Agent:", result)
