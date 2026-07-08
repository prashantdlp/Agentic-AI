from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma2:2b")

class AgentState(TypedDict):
	question: str
	answer: str
	score: int
	iterations: int

def generate(state):
	print("Entered generate")
	prompt = f"Answer: {state['question']}"
	response = llm.invoke(prompt)
	return {
		"answer": response.content,
		"iterations": state["iterations"] + 1
	}

def evaluate(state):
	print("Entered evaluate")
	answer = state["answer"]
	words = len(answer.split())

	score = state["score"]
	if words > 100:
		score += 1

	return {"score": score}

def refine(state):
	print("Entered refine")
	response = llm.invoke(f"Improve this:\n{state['answer']}")
	return {"answer": response.content}

def route(state):
	if state["score"] >= 1:
		print("Routed to end")
		return "end"
	print("Routed to refine")
	return "refine"

builder = StateGraph(AgentState)

builder.add_node("generate", generate)
builder.add_node("evaluate", evaluate)
builder.add_node("refine", refine)

builder.set_entry_point("generate")

builder.add_edge("generate", "evaluate")
builder.add_edge("refine", "evaluate")

builder.add_conditional_edges(
    "evaluate",
    route,
    {
        "end": END,
        "refine": "refine"
    }
)

graph = builder.compile()

query = input("Enter your topic: ")
initial_state = {
    "question": query,
    "answer": "",
    "score": 0,
    "iterations": 0
}

result = graph.invoke(initial_state)
print(result)
