from src.Langgraph_Agentic_AI.state.state import State

class Basic_Chatbot_Node:
    """
    Basic Chatbot Login Implementation
    """
    def __init__(self,model):
        self.llm = model
        
    def process(self, state: State) -> dict:
        """
        Processes the input state and generates a chatbot response.
        """
        return {"messages": self.llm.invoke(state["messages"])}