import asyncio
import os
import subprocess
from enum import Enum  
from typing import List, Literal, TypedDict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment variables.")

planner_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)  
review_llm  = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
error_llm   = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY) 

class ErrorType(str, Enum):
    MISSING_PACKAGE = "MISSING_PACKAGE"
    COMMAND_NOT_FOUND = "COMMAND_NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NETWORK_ERROR = "NETWORK_ERROR"
    VERSION_MISMATCH = "VERSION_MISMATCH"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    UNKNOWN = "UNKNOWN"

class PlannerState(TypedDict):
    plan: List[str]

class ReviewerState(TypedDict):
    feedback: str
    approved: bool
    revision_count: int

class ExecutorState(TypedDict):
    completed_steps: List[str]
    current_output: str
    failed_command: Optional[str]      
    error_type: Optional[str]          

class SharedState(TypedDict):
    task: str
    iterations: int
    error_recovery_count: int         
    logs: List[str]

class GlobalState(TypedDict):
    shared: SharedState
    planner: PlannerState
    reviewer: ReviewerState
    executor: ExecutorState

class PlannedCommands(BaseModel):
    commands: List[str] = Field(description="Sequential list of executable bash commands.")

class ReviewResult(BaseModel):
    approved: bool = Field(description="True if the plan is completely safe and effective, False otherwise.")
    feedback: str = Field(description="Constructive critique if rejected, or confirmation if approved.")

class ErrorAnalysis(BaseModel):
    error_type: ErrorType = Field(description="The categorized type of shell execution error.")
    explanation: str = Field(description="Explanation of why it failed and explicit guidance on how the planner can fix it.")

def planner_node(state: GlobalState) -> dict:
    print("\n[Planner]: Formulating environment management plan...")
    
    feedback_str = ""
    if state["reviewer"]["feedback"]:
        feedback_str += f"\nReviewer Feedback: {state['reviewer']['feedback']}"
    if state["executor"]["current_output"] and "Error" in state["executor"]["current_output"]:
        feedback_str += f"\nPrevious Execution Error on command '{state['executor']['failed_command']}': {state['executor']['current_output']}"

    prompt = f"""
    You are a Development Environment Assistant. Suggest sequential bash commands to achieve the given task.
    
    Task: {state["shared"]["task"]}
    Context/Feedback (if any): {feedback_str}

    Constraints:
    - Only return actionable, valid bash commands.
    - Adjust your plan dynamically based on any previous execution errors or reviewer feedback provided above.
    - Do not suggest destructive commands (e.g., rm -rf /).
    - If you need to check things (e.g., python version), make that the first command.
    """
    
    structured_planner = planner_llm.with_structured_output(PlannedCommands)
    response = structured_planner.invoke(prompt)

    print("[Planner] New Proposed Commands: ", response.commands)

    new_logs = state["shared"]["logs"] + ["Planner updated the command execution list."]
    
    return {
        "planner": {"plan": response.commands},
        "shared": {**state["shared"], "logs": new_logs}
    }

def reviewer_node(state: GlobalState) -> dict:
    print("[Reviewer]: Scanning plan safety and validity...")
    
    prompt = f"""
    You are an Environment Security and Setup Expert. Review the proposed bash commands for safety and correctness.
    
    Target Task: {state["shared"]["task"]}
    Proposed Plan: {state["planner"]["plan"]}
    
    Reject any plans that contain dangerous parameters, loops that could hang, or syntax errors.
    """
    
    structured_reviewer = review_llm.with_structured_output(ReviewResult)
    response = structured_reviewer.invoke(prompt)
    
    current_revisions = state["reviewer"]["revision_count"] + 1
    new_logs = state["shared"]["logs"] + [f"Reviewer pass #{current_revisions}: Approved={response.approved}"]
    
    return {
        "reviewer": {
            "feedback": response.feedback,
            "approved": response.approved,
            "revision_count": current_revisions
        },
        "shared": {**state["shared"], "logs": new_logs}
    }

def executor_node(state: GlobalState) -> dict:
    plan = state["planner"]["plan"]
    completed = state["executor"]["completed_steps"]
    next_step_index = len(completed)
    
    if next_step_index >= len(plan):
        return {} 
        
    command_to_run = plan[next_step_index]
    print(f"[Executor]: Running command: {command_to_run}")
    
    failed_command = None
    error_type = None
    
    try:
        result = subprocess.run(command_to_run, shell=True, text=True, capture_output=True, timeout=30)
        if result.returncode == 0:
            output = f"Success:\n{result.stdout}"
            completed.append(command_to_run)
        else:
            output = f"Error (Exit Code {result.returncode}):\n{result.stderr}"
            failed_command = command_to_run
    except Exception as e:
        output = f"Execution Exception: {str(e)}"
        failed_command = command_to_run

    print(f"[Execution Result]: {output.strip()}")
    
    new_logs = state["shared"]["logs"] + [f"Executed: {command_to_run}"]
    return {
        "executor": {
            "completed_steps": completed,
            "current_output": output,
            "failed_command": failed_command,
            "error_type": error_type
        },
        "shared": {**state["shared"], "logs": new_logs}
    }

def error_handler_node(state: GlobalState) -> dict:
    """New node tasked with classifying runtime errors and structuring a correction strategy."""
    print("[Error Handler]: Analyzing runtime crash details...")
    
    failed_cmd = state["executor"]["failed_command"]
    raw_error = state["executor"]["current_output"]
    
    prompt = f"""
    Analyze the following terminal command execution failure:
    Command attempted: {failed_cmd}
    Terminal Output: {raw_error}
    
    Classify the error type and write clear, instructive feedback detailing how to resolve this.
    """
    
    structured_analyzer = error_llm.with_structured_output(ErrorAnalysis)
    analysis = structured_analyzer.invoke(prompt)
    
    print(f"[Error Breakdown]: Category={analysis.error_type.value} | Guide={analysis.explanation}")
    
    current_recoveries = state["shared"]["error_recovery_count"] + 1
    new_logs = state["shared"]["logs"] + [f"Error Handled ({analysis.error_type.value}). Attempting adjustment."]
    
    return {
        "executor": {
            **state["executor"],
            "error_type": analysis.error_type.value
        },
        "reviewer": {
            **state["reviewer"],
            "feedback": f"CRITICAL RUNTIME ERROR [{analysis.error_type.value}]: {analysis.explanation}",
            "approved": False  # Force a replan pass
        },
        "shared": {
            **state["shared"],
            "error_recovery_count": current_recoveries,
            "logs": new_logs
        }
    }

def route_after_review(state: GlobalState) -> Literal["executor", "planner", "end"]:
    if state["reviewer"]["approved"]:
        return "executor"
    
    if state["reviewer"]["revision_count"] >= 3:
        print("[System]: Revision limit reached without approval. Terminating workflow safely.")
        return "end"
        
    return "planner"

def route_after_execution(state: GlobalState) -> Literal["executor", "error_handler", "end"]:
    # Reroute to error handler if a command failed
    if state["executor"]["failed_command"] is not None:
        if state["shared"]["error_recovery_count"] >= 2:
            print("[System]: Self-healing failed after consecutive execution attempts. Aborting safely.")
            return "end"
        print("[System]: Execution failure detected. Passing context to Error Handler...")
        return "error_handler"

    completed = len(state["executor"]["completed_steps"])
    total = len(state["planner"]["plan"])

    if completed >= total:
        print("[System]: All planned environment updates completed successfully.")
        return "end"
        
    return "executor"

builder = StateGraph(GlobalState)

builder.add_node("planner", planner_node)
builder.add_node("reviewer", reviewer_node)
builder.add_node("executor", executor_node)
builder.add_node("error_handler", error_handler_node) 

builder.set_entry_point("planner")
builder.add_edge("planner", "reviewer")
builder.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "executor": "executor",
        "planner": "planner",
        "end": END
    }
)
builder.add_conditional_edges(
    "executor",
    route_after_execution,
    {
        "executor": "executor",
        "error_handler": "error_handler",
        "end": END
    }
)
builder.add_edge("error_handler", "planner")

graph = builder.compile()

if __name__ == "__main__":
    query = input("Enter your Development Environment Related Query: ")
    
    initial_state = {
        "shared": {
            "task": query,
            "iterations": 0,
            "error_recovery_count": 0,
            "logs": []
        },
        "planner": {
            "plan": []
        },
        "reviewer": {
            "feedback": "",
            "approved": False,
            "revision_count": 0
        },
        "executor": {
            "completed_steps": [],
            "current_output": "",
            "failed_command": None,
            "error_type": None
        }
    }
    
    final_output = graph.invoke(initial_state)
    print("\n--- Final Workflow Logs ---")
    for log in final_output["shared"]["logs"]:
        print(f"- {log}")