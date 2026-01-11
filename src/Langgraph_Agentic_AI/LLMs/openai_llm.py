import os
import streamlit as st
from langchain_openai import ChatOpenAI

class OpenAILLM:
    def __init__(self, user_controls_input):
        self.user_controls_input = user_controls_input

    def get_llm(self):
        openai_api_key = self.user_controls_input.get("OPENAI_API_KEY")
        selected_openai_model = self.user_controls_input.get("selected_openai_model")

        if not openai_api_key:
            st.error("Please enter OPENAI API KEY")
            return None

        try:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            llm = ChatOpenAI(model=selected_openai_model)
            return llm
        except Exception as e:
            raise ValueError(f"OpenAI LLM error: {e}")