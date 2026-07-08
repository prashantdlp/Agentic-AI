from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
	{
	    "gmail": {
		"command": "npx",
		"args": ["@gongrzhe/server-gmail-autoauth-mcp"],
		"transport": "stdio",
	    }
	}
)

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq # fast
# from langchain_ollama import ChatOllama # slow af
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile") #, api_key = GROQ_API_KEY)

class AgentState(TypedDict):
	context: str # recipient name & email, sender name & email, context string
	draft: str
	score: int # heuristic for exiting the loop
	iterations: int

def generate(state: AgentState):

	prompt = f"""
		Write a professional cold email based on the context below.

		Context:{state["context"]}

		Sender Name: Prashant Meena

		Requirements:
		- Write only the final email body
		- No placeholders (like [Name], YOUR_NAME, COMPANY)
		- No notes, explanations, or markdown
		- Concise, personalized, and professional
		- Sound natural and human, not robotic
		- Clear reason for outreach
		- End with a simple call-to-action
		- Sign off with: Prashant Meena

		This email will be sent exactly as generated, so make it fully ready to send.
	"""

	response = llm.invoke(prompt)

	return {
		"draft": response.content
	}

def review(state: AgentState):
	prompt = f"""
	Review this email and score it from 1-10.

	Email:
	{state["draft"]}

	Score based on:
	- clarity
	- professionalism
	- grammar
	- call to action

	Return ONLY integer score.
	"""

	response = llm.invoke(prompt)

	try:
		score = int(response.content.strip())
	except:
		score = 5

	return {
		"score": score,
		"iterations": state["iterations"] + 1
	}

def refine(state: AgentState):
	prompt = f"""
	Improve this email to get score 9+.

	Current email:
	{state["draft"]}
	"""

	response = llm.invoke(prompt)

	return {
		"draft": response.content
	}

def route(state: AgentState):
	if state["score"] >= 8:
		return "end"

	if state["iterations"] >= 3:
		return "end"

	return "refine"

# initial -> generate -> review -> end($)
#		^	    |
#	 	|	    |
# 		<------------

builder = StateGraph(AgentState)

builder.add_node("generate", generate)
builder.add_node("review", review)
builder.add_node("refine", refine)

builder.set_entry_point("generate")

builder.add_edge("generate", "review")
builder.add_edge("refine", "review")

builder.add_conditional_edges(
	"review",
	route,
	{
		"end": END,
		"refine": "refine"
	}
)

graph = builder.compile()

# sender = input(), we already know
recipient_name = input("Enter Recipient's Name: ")
recipient_email = input("Enter Recipient's Valid Gmail: ")
email_context = input("Provide some context regarding Email: ")


initial_state = {
	"context": f"""
		recipient_name: {recipient_name},
		recipient_email: {recipient_email},
		email_context:{email_context}
	""",
	"draft": "",
	"score": 0,
	"iterations": 0
}
result = graph.invoke(initial_state)

async def send(final_draft: str): # USES MCP SERVER
	global recipient_name, recipient_email

	tools = await client.get_tools()

	print("Available tools:")
	for tool in tools:
		print(tool.name)

	for tool in tools:
		if "send" in tool.name.lower():
			response = await tool.ainvoke({
				"to": [recipient_email],
				"subject": "Quick Introduction",
				"body": final_draft
			})
			print(response)
			return

	print("No send tool found.")

asyncio.run(send(result["draft"]))












