import os
import pandas as pd
from typing import TypedDict, Annotated, List, Optional, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_aws import ChatBedrock  # or langchain_openai.ChatOpenAI, etc.
from dotenv import load_dotenv

# ============================================
# STEP 1: ENVIRONMENT SETUP
# ============================================
load_dotenv()
MODEL_ID = os.getenv("CHATBOT_LLM_ID")  # Your model ID
REGION = os.getenv("REGION")            # Your region (if AWS)
ALL_ROLES = ["TOP", "MIDDLE", "JUNGLE", "BOTTOM", "UTILITY"]
ALL_QUEUES = ["RANKED", "DRAFT", "BOTH", "AVERAGE_CHAMPION_X"]
USER_CHAMPION_SELECTION = ["MOST_PLAYED", "HIGHEST_WR", "MANUAL"]
EXPLORE_OR_OPTIMIZE = ["EXPLORATION", "OPTIMIZATION"]
OPTIMIZATION_SCOPE = ["CLUSTER", "ROLE"]

# ============================================
# STEP 2: STATE DEFINITION
# ============================================
# This is your "memory" that gets passed between nodes
# Think of it as a shared notebook that all nodes can read/write

class GraphState(TypedDict):
    # REQUIRED: For conversation history
    messages: Annotated[List[BaseMessage], add_messages]
    current_step: str
    # ============================================
    # MAIN_GRAPH: First branch of nodes
    # ============================================
    use_own_data: Optional[bool] # [LLM_NODE] 
    user_selected_role: Optional[str] # [LLM_NODE] needs to be one out of the 5 in ALL_ROLES
    user_selected_queue_type: Optional[str] # [LLM_NODE] needs to be one out of the 4 in ALL_QUEUES
    user_champion_selection: Optional[str] # [LLM_NODE] needs to be one out of the 3 in USER_CHAMPION_SELECTION - min 10 games!
    filtered_user_data: Optional[pd.DataFrame] # [REC_ACTION] If not enough games, ask again 
    user_champion_cluster_description: Optional[pd.DataFrame] # [REC_ACTION]
    explore_or_optimize: Optional[str] # [LLM_NODE] needs to be one out of the 2 in EXPLORE_OR_OPTIMIZE
    
    # ============================================
    # SUB_GRAPH_1: Optimization
    # ============================================
    user_desired_scope: Optional[str] # [LLM_NODE] needs to be one out of the 2 in OPTIMIZATION_SCOPE
    df_filtered_by_scope: Optional[pd.DataFrame] # [REC_ACTION] Filter data by cluster or only role
    win_rate_criteria: Optional[bool] # [LLM_NODE] want to round out champ pool based on win rates?
    # ============================================
    # SUB_GRAPH_1_1: Optimization - "YES win rate criteria"
    # ============================================
    win_rate_optimized_champ_rec: Optional[str] # [REC_ACTION] 
    better_blind_vs_og_champ: Optional[bool] # [REC_ACTION] 
    best_ban_rec: Optional[str] # [REC_ACTION]
    recommendation_1_1: Optional[str] # [LLM_FINAL_NODE]
    # ============================================
    # SUB_GRAPH_1_2: Optimization - "NO win rate criteria"
    # ============================================
    clusters_most_similar_champ: Optional[str] # [REC_ACTION]
    clusters_most_different_champ: Optional[str] # [REC_ACTION]
    clusters_similar_players_champ: Optional[str] # [REC_ACTION]
    clusters_highest_wr_champ: Optional[str] # [REC_ACTION]
    recommendation_1_2: Optional[str] # [LLM_FINAL_NODE]
    # ============================================
    # SUB_GRAPH_2: Exploration
    # ============================================
    remaining_clusters_descriptions: Optional[dict[str]] # [REC_ACTION]
    preferred_cluster: Optional[str] # [LLM_NODE] 
    pref_cluster_champions: Optional[pd.DataFrame] # [REC_ACTION]
    similar_champions_to_og_champ: Optional[bool] # [LLM_NODE]
    # ============================================
    # SUB_GRAPH_2_1: Exploration - "YES similarity criteria"
    # ============================================
    similar_champs_with_descriptions_and_tags: Optional[dict] # [REC_ACTION] DataFrame with semantic tags and description of 3 champions
    chosen_champion_based_on_description_and_tags: Optional[str] # [LLM_NODE]
    recommendation_2_1: Optional[str] # [LLM_FINAL_NODE]
    # ============================================
    # SUB_GRAPH_2_1: Exploration - "NO similarity criteria"
    # ============================================
    top_representative_champions_descriptions_and_tags: Optional[pd.DataFrame] # [REC_ACTION] top 5 champions most representative of cluster with descriptions and tags
    preferred_rep_champion: Optional[str] # [LLM_NODE] Did you like any of the 5?
    # ============================================
    # SUB_GRAPH_2_1_1: Exploration - "DID like a representative champions"
    # ============================================
    recommendation_2_1_1: Optional[str] # [LLM_FINAL_NODE] Recommend preferred champion
    # ============================================
    # SUB_GRAPH_2_1_2: Exploration - "DID NOT like a representative champions"
    # ============================================
    recommendation_2_1_2: Optional[str] # [LLM_FINAL_NODE] Provide highest win rate champion outside of the 5 representative ones

# ============================================
# STEP 3: LLM SETUP
# ============================================
# Initialize your LLM (change based on your provider)
llm = ChatBedrock(
    model_id=MODEL_ID,
    region=REGION,
    temperature=0.5  # Adjust for creativity vs consistency
)

# ============================================
# STEP 4: NODE DEFINITIONS
# ============================================
# Nodes are the boxes in your diagram. Each node is a function that:
# - Takes the current state as input
# - Does something (ask question, process data, make decision)
# - Returns updates to the state

# ---------- QUESTION NODES (Rectangles) ----------
# These nodes ask the user something

def ask_use_own_data(state: GraphState) -> GraphState:
    """
    Asks user if they want to use their own data.
    """
    question = "Would you like to use your own data for our analysis?"
    
    # Create an AI message with the question
    ai_message = AIMessage(content=question)
    
    # Return state updates (messages will be appended due to add_messages)
    return {
        "messages": [ai_message],
        "current_step": "asked_use_own_data"
    }

def ask_desired_lane(state: GraphState) -> GraphState:
    """Another question node"""
    # You can access previous state data
    previous_choice = state.get("user_choice_1", "unknown")
    
    question = f"Based on your choice of {previous_choice}, would you like feature X? (yes/no)"
    
    return {
        "messages": [AIMessage(content=question)],
        "current_step": "asked_followup"
    }

# ---------- PROCESSING NODES (Circles) ----------
# These nodes process user responses or perform actions

def process_user_response(state: GraphState) -> GraphState:
    """
    Example of processing the user's last message.
    This is where you extract/parse what the user said.
    """
    # Get the last user message
    last_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_message = msg.content
            break
    
    # Parse the response (simple example - you might use LLM here)
    if last_message:
        if "option a" in last_message.lower():
            user_choice = "A"
        elif "option b" in last_message.lower():
            user_choice = "B"
        else:
            user_choice = "unknown"
    else:
        user_choice = "no_response"
    
    # Update state with parsed information
    return {
        "user_choice_1": user_choice,
        "current_step": "processed_initial"
    }

def fetch_data_node(state: GraphState) -> GraphState:
    """
    Example of a node that fetches/computes data.
    Replace with your actual data fetching logic.
    """
    # Placeholder for your data fetching
    # data = your_api_call() 
    # data = your_database_query()
    # data = your_ml_model_prediction()
    
    fake_data = {
        "result": "some_computed_value",
        "score": 0.95
    }
    
    return {
        "some_data": fake_data,
        "current_step": "data_fetched"
    }

def provide_recommendation(state: GraphState) -> GraphState:
    """
    Example of a final node that provides output to user.
    """
    # Use data from state to create recommendation
    user_choice = state.get("user_choice_1", "unknown")
    data = state.get("some_data", {})
    
    recommendation = f"Based on your choice of {user_choice}, I recommend: {data.get('result', 'N/A')}"
    
    return {
        "messages": [AIMessage(content=recommendation)],
        "current_step": "complete"
    }

# ============================================
# STEP 5: CONDITIONAL ROUTING FUNCTIONS
# ============================================
# These determine which node to go to next based on state

def route_after_initial(state: GraphState) -> Literal["path_a", "path_b", "error"]:
    """
    Routing logic after processing initial response.
    Returns the name of the next node to visit.
    """
    choice = state.get("user_choice_1", "unknown")
    
    if choice == "A":
        return "path_a"
    elif choice == "B":
        return "path_b"
    else:
        return "error"

def route_yes_no(state: GraphState) -> Literal["yes_path", "no_path"]:
    """
    Another routing function for yes/no decisions.
    """
    # Check the last message for yes/no
    last_message = state["messages"][-1] if state["messages"] else None
    
    if last_message and isinstance(last_message, HumanMessage):
        if "yes" in last_message.content.lower():
            return "yes_path"
    
    return "no_path"

# ============================================
# STEP 6: BUILD THE GRAPH
# ============================================

# Initialize the graph with your state type
workflow = StateGraph(GraphState)

# ---------- ADD ALL YOUR NODES ----------
# workflow.add_node("node_name", node_function)

workflow.add_node("ask_initial", ask_use_own_data)
workflow.add_node("process_initial", process_user_response)
workflow.add_node("ask_followup", ask_desired_lane)
workflow.add_node("fetch_data", fetch_data_node)
workflow.add_node("recommend", provide_recommendation)

# You can add an error handler node
workflow.add_node("error", lambda state: {
    "messages": [AIMessage(content="I didn't understand that. Let's try again.")],
    "current_step": "error"
})

# ---------- CONNECT THE NODES WITH EDGES ----------

# Simple edges (always go from A to B)
workflow.add_edge(START, "ask_initial")
workflow.add_edge("ask_initial", "process_initial")  # After asking, process response

# Conditional edges (go to different nodes based on logic)
workflow.add_conditional_edges(
    "process_initial",  # From this node...
    route_after_initial,  # Use this function to decide...
    {
        "path_a": "ask_followup",  # If function returns "path_a", go here
        "path_b": "fetch_data",     # If function returns "path_b", go here  
        "error": "error"            # If function returns "error", go here
    }
)

# More edges to complete the flow
workflow.add_edge("ask_followup", "fetch_data")
workflow.add_edge("fetch_data", "recommend")
workflow.add_edge("recommend", END)
workflow.add_edge("error", "ask_initial")  # Loop back on error

# ============================================
# STEP 7: COMPILE THE GRAPH
# ============================================
app = workflow.compile()

# ============================================
# STEP 8: RUN THE GRAPH
# ============================================
if __name__ == "__main__":
    # Initialize with empty state
    initial_state = {
        "messages": [],
        "current_step": "start"
    }
    
    # Method 1: Run the whole graph once
    final_state = app.invoke(initial_state)
    print("Final state:", final_state)
    
    # Method 2: Interactive mode (for chatbots)
    # This requires handling user input between invocations
    """
    current_state = initial_state
    
    while True:
        # Run graph until it needs user input
        result = app.invoke(current_state)
        
        # Show the last AI message to user
        last_ai_message = result["messages"][-1]
        print("AI:", last_ai_message.content)
        
        # Check if we're done
        if result.get("current_step") == "complete":
            break
            
        # Get user input
        user_input = input("You: ")
        
        # Add user message to state and continue
        current_state = result
        current_state["messages"].append(HumanMessage(content=user_input))
    """

# ============================================
# COMMON PATTERNS & TIPS
# ============================================
"""
1. ASK-PARSE PATTERN:
   ask_question → get_user_response → parse_response → route_based_on_parse
   
2. LLM PARSING (more robust than keyword matching):
   def parse_with_llm(user_text: str) -> dict:
       prompt = "Extract the user's intent from: {text}"
       response = llm.invoke(prompt.format(text=user_text))
       # Parse LLM response to structured data
       return parsed_data

3. SUBPROCESS PATTERN:
   You can create sub-graphs and call them from main graph
   
4. PARALLEL EXECUTION:
   Use add_conditional_edges to create parallel paths that merge later

5. STATE PERSISTENCE:
   You can save/load state to database between sessions

6. ERROR HANDLING:
   Always have fallback paths for unexpected user inputs

7. DEBUGGING:
   Add "current_step" to state to track where you are in the flow
   Use print statements in nodes during development

8. TESTING:
   Test each node function independently before connecting them
"""