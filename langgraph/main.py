from typing import Literal
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


@tool
def _search(query: str):
    """Call to surf the web."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


_tools = [_search]
_tool_node = ToolNode(_tools)
_model = ChatOpenAI(model="gpt-4o-mini").bind_tools(_tools)


def _should_continue(state: MessagesState) -> Literal["tools", END]:
    msgs = state["messages"]
    last_msg = msgs[-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


def _call_model(state: MessagesState):
    msgs = state["messages"]
    resp = _model.invoke(msgs)
    return {"messages": [resp]}


workflow = StateGraph(MessagesState)
workflow.add_node("agent", _call_model)
workflow.add_node("tools", _tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", _should_continue)
workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)

final_state = app.invoke(
    {"messages": HumanMessage("What is the weather in sf?")},
    config={"configurable": {"thread_id": "ewrafafyuewrsdf"}},
)
print(final_state["messages"][-1])

final_state = app.invoke(
    {"messages": HumanMessage("what about ny?")},
    config={"configurable": {"thread_id": "ewrafafyuewrsdf"}},
)
print(final_state["messages"][-1])
