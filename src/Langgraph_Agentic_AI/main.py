import streamlit as st
from src.Langgraph_Agentic_AI.ui.streamlit.load_ui import LoadStreamlitUI
from src.Langgraph_Agentic_AI.LLMs.groq_llm import GroqLLM
from src.Langgraph_Agentic_AI.LLMs.openai_llm import OpenAILLM
from src.Langgraph_Agentic_AI.graph.graph_builder import Graph_Builder
from src.Langgraph_Agentic_AI.ui.streamlit.display_result import DisplayResultStreamlit

def load_langgraph_agenticai_app():
    """
    Loads and runs the LangGraph Agentic AI application with Streamlit UI.
    This function initializes the UI, handles user input, configures the LLM model,
    sets up the graph based on the selected use case, and displays the output while
    implementing exception handling for robustness.
    """
    
    ## Load UI
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()
    
    if not user_input:
        st.error("Error: Failed to load user input from the UI")
        return
    
    ## Text Input for user message
    if st.session_state.IsFetchButtonClicked:
        user_message = st.session_state.timeframe
    else: 
        user_message = st.chat_input("Enter you message: ")
    
    if not user_message:
        st.info("Please enter a message to continue.")
        return

    # Configuring LLMs
    model = None
    if user_input["selected_llm"] == "Groq":
        obj_llm_config = GroqLLM(user_controls_input=user_input)
        model = obj_llm_config.get_llm()
    elif user_input["selected_llm"] == "OpenAI":
        obj_llm_config = OpenAILLM(user_controls_input=user_input)
        model = obj_llm_config.get_llm()
    
    if not model:
        st.error("Error: LLM could not be initialized")
        return

    # --- Check selected use case ---
    usecase = user_input.get("selected_usecase")
    if not usecase:
        st.error("Error: No use case selected")
        return

    # --- Build Graph ---
    try:
        graph_builder = Graph_Builder(model)
        graph = graph_builder.setup_graph(usecase)
        DisplayResultStreamlit(usecase, graph, user_message).display_result_on_ui()
    except Exception as e:
        st.error(f"Error: Graph Setup failed: {e}")
        return
