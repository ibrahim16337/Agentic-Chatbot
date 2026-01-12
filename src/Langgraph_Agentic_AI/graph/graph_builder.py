from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from src.Langgraph_Agentic_AI.state.state import State
from src.Langgraph_Agentic_AI.nodes.basic_chatbot_node import Basic_Chatbot_Node
from src.Langgraph_Agentic_AI.tools.search_tool import get_tools, create_tool_node
from src.Langgraph_Agentic_AI.nodes.chabot_with_tool_node import Chatbot_with_Tool_Node
from src.Langgraph_Agentic_AI.nodes.ai_news_node import AI_News_Node


class Graph_Builder:
    def __init__(self, model):
        self.llm = model
        self.graph_builder = StateGraph(State)
        
    def basic_chatbot_graph_builder(self):
        """ 
        Builds a basic chatbot graph using LangGraph.
        This method initializes a chatbot node using the class
        and integrates it into the graph. The chatbot node is set as both the
        entry and exit point of the graph.
        """
        
        self.basic_chatbot_node = Basic_Chatbot_Node(self.llm)
        
        self.graph_builder.add_node("chatbot", self.basic_chatbot_node.process)
        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_edge("chatbot", END)
        
    def chatbot_with_tools_graph_builder(self):
        """
        Builds an advanced chatbot graph with tool integration.
        This method creates a chatbot graph that includes both a chatbot node
        and a tool node. It defines tools, initializes the chatbot with tool
        capabilities, and sets up conditional and direct edges between nodes.
        The chatbot node is set as the entry point.
        """
        ## Define the tool and tool node
        tools = get_tools()
        tool_node = create_tool_node(tools)
        
        ## Define the LLM
        llm = self.llm
        
        ## Define the chatbot node
        obj_chatbot_with_tool = Chatbot_with_Tool_Node(llm)
        chatbot_node = obj_chatbot_with_tool.create_chatbot(tools)
        
        ## Add Nodes
        self.graph_builder.add_node("chatbot", chatbot_node)
        self.graph_builder.add_node("tools", tool_node)
        
        ## Add Edges
        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_conditional_edges("chatbot", tools_condition)
        self.graph_builder.add_edge("tools", "chatbot")
        self.graph_builder.add_edge("chatbot", END)
        
    def ai_news_graph_builder(self):
        ## Define th AI News Node
        ai_news_node = AI_News_Node(self.llm)
        
        ## Add Nodes
        self.graph_builder.add_node("fetch_news", ai_news_node.fetch_news)
        self.graph_builder.add_node("summarize_news", ai_news_node.summarize_news)
        self.graph_builder.add_node("save_result", ai_news_node.save_result)
        
        ## Add Edges
        self.graph_builder.set_entry_point("fetch_news")
        self.graph_builder.add_edge("fetch_news", "summarize_news")
        self.graph_builder.add_edge("summarize_news", "save_result")
        self.graph_builder.add_edge("save_result", END)
    
    def setup_graph(self, usecase: str):
        """
        Sets up the graph for the selected use case.
        """
        if usecase == "Basic Chatbot":
            self.basic_chatbot_graph_builder()
        if usecase == "Chatbot with Web Search":
            self.chatbot_with_tools_graph_builder()
        if usecase == "AI News":
            self.ai_news_graph_builder()   
            
            
        return self.graph_builder.compile()
    
    