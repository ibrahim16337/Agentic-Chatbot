from src.Langgraph_Agentic_AI.state.state import State

class Chatbot_with_Tool_Node:
    """
    Chatbot logic enhaced with tool integration
    """
    def __init__(self, model):
        self.llm = model
        
    def process(self, state: State) -> dict:
        """
        Processes the input state and generaes a response with tool integration
        """
        user_input = state["messages"][-1] if state["messages"] else ""
        llm_response = self.llm.invoke([{"role": "user", "content": user_input}])
        
        ## Simulate tool specific logic
        tool_response = f"Tool integration for: '{user_input}'"
        return {"messages": [llm_response, tool_response]}
        
    def create_chatbot(self, tools):
        """
        Returns a chatbot node function
        """
        llm_with_tools = self.llm.bind_tools(tools)
        
        def chatbot_node(state: State):
            """
            Chatbot logic for precessing the input state and returning a response
            """
            return {"messages": [llm_with_tools.invoke(state["messages"])]}
        
        return chatbot_node