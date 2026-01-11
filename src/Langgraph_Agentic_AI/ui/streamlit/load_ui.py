import streamlit as st
import os

from src.Langgraph_Agentic_AI.ui.uiconfigfile import Config

class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}
        
    def load_streamlit_ui(self):
        st.set_page_config(page_title = self.config.get_page_title(), layout="wide")
        st.header(self.config.get_page_title())
        
        with st.sidebar:
            ## Get Options from Config
            llm_options = self.config.get_llm_options()
            usecase_options = self.config.get_usecase_options()
            
            ## LLM Selection
            self.user_controls['selected_llm'] = st.selectbox("Select LLM", llm_options)
            
            if self.user_controls["selected_llm"] == "Groq":
                ## Model Selection
                model_options = self.config.get_groq_model_options()
                self.user_controls["selected_groq_model"] = st.selectbox("Select Model", model_options)
                self.user_controls["GROQ_API_KEY"] = st.session_state["GROQ_API_KEY"] = st.text_input("API Key", type="password")
                if not self.user_controls["GROQ_API_KEY"]:
                    st.warning("Please Enter you GROQ API KEY to proceed. Dont have? refer: https://console.groq.com/keys")
            elif self.user_controls["selected_llm"] == "OpenAI":
                ## Model Selection
                model_options = self.config.get_openai_model_options()
                self.user_controls["selected_openai_model"] = st.selectbox("Select Model", model_options)
                self.user_controls["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"] = st.text_input("API Key", type="password")
                if not self.user_controls["OPENAI_API_KEY"]:
                    st.warning("Please Enter you OPENAI API KEY to proceed. Dont have? refer: https://platform.openai.com/settings/organization/api-keys")
            
            ## Usecase Selection
            self.user_controls["selected_usecase"] = st.selectbox("Select Usecase", usecase_options)
            
        return self.user_controls