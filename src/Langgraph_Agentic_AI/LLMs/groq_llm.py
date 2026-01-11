import os
import streamlit as st
from langchain_groq import ChatGroq

class GroqLLM:
    def __init__(self, user_controls_input):
        self.user_controls_input = user_controls_input

    def get_llm(self):
        groq_api_key = self.user_controls_input.get("GROQ_API_KEY")
        selected_groq_model = self.user_controls_input.get("selected_groq_model")

        if not groq_api_key:
            st.error("Please enter GROQ API KEY")
            return None

        try:
            os.environ["GROQ_API_KEY"] = groq_api_key
            llm = ChatGroq(model=selected_groq_model)
            return llm
        except Exception as e:
            raise ValueError(f"Groq LLM error: {e}")
