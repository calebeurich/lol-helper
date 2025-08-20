import os
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_aws import ChatBedrock
from dotenv import load_dotenv

load_dotenv()
MODEL_ID = os.getenv("CHATBOT_LLM_ID")
REGION = os.getenv("REGION")

# State
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: Optional[str]

# LLM config
llm = ChatBedrock(
    model_id=MODEL_ID,
    region=REGION,
    temperature=0.2
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use context if provided.\n\n<context>\n{context}\n</context>"),
    MessagesPlaceholder(variable_name="messages"),
])

# First node
def chat_node(state: ChatState) -> ChatState:
    chain = prompt | llm
    ai = chain.invoke({"messages": state["messages"], "context": state.get("context", "")})
    return {"messages": [ai]}  # add_messages will append

# Graph: single turn per invoke
workflow = StateGraph(ChatState)
workflow.add_node("chat", chat_node)
workflow.add_edge(START, "chat")
workflow.add_edge("chat", END)
app = workflow.compile()