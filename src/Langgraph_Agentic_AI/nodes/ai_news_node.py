# from tavily import TavilyClient
# from langchain_core.prompts import ChatPromptTemplate

# class AI_News_Node:
#     def __init__(self, llm):
#         """
#         Initialize the AI News Node with API Keys for Tavily and OpenAI
#         """
#         self.tavily = TavilyClient()
#         self.llm = llm
        
#         ## This is used to capture various steps in this file and later can be used to show steps
#         self.state = {}
        
#     # def fetch_news(self, state: dict) -> dict:
#     #     """
#     #     Fetch AI news based on the specified frequency.
#     #     Args :
#     #         state (dict): The state dictionary containing 'frequency' .
#     #     Returns :
#     #         dict: Updated state with 'news_data' key containing fetched news.
#     #     """
        
#     #     frequency = state["messages"][0].content.lower()
#     #     self.state["frequency"] = frequency
#     #     time_range_map = {"daily": "d", "weekly": "w", "monthly": "m", "year": "y"}
#     #     days_map = {"daily": 1, "weekly": 7, "monthly": 30, "year": 366}
        
#     #     response = self.tavily.search(
#     #         query = "Top Artificial Intelligence (AI) technology news Pakistan and globally",
#     #         topic = "news",
#     #         time_range = time_range_map[frequency],
#     #         include_answer = "advanced",
#     #         max_results = 15,
#     #         days = days_map[frequency],
#     #     )
        
#     #     state["news_data"] = response.get("results", [])
#     #     self.state["news_data"] = state["news_data"]
#     #     return state
    
#     def fetch_news(self, state: dict) -> dict:
#         raw_text = ""

#         for msg in state.get("messages", []):
#             if hasattr(msg, "content") and msg.content:
#                 raw_text = msg.content.lower().strip()
#                 break

#         allowed_frequencies = ["daily", "weekly", "monthly", "year"]
#         frequency = next(
#             (f for f in allowed_frequencies if f in raw_text),
#             "daily"
#         )

#         self.state["frequency"] = frequency

#         days_map = {"daily": 1, "weekly": 7, "monthly": 30, "year": 366,}

#         response = self.tavily.search(
#             query="Top Artificial Intelligence (AI) technology news Pakistan and globally",
#             topic="news",
#             include_answer="advanced",
#             max_results=15,
#             days=days_map[frequency],
#         )

#         state["news_data"] = response.get("results", [])
#         self.state["news_data"] = state["news_data"]

#         return state


#     def summarize_news(self, state: dict) -> dict:
#         """ 
#         Summarize the fetched news using an LLM.
#         Args :
#             state (dict): The state dictionary containing 'news _ data' .
#         Returns :
#             dict: Updated state with 'summary' key containing the summarized news.
#         """
#         news_items = self.state["news_data"]
        
#         prompt_template = ChatPromptTemplate([
#             ("system", """Summarize AI Newsarticles into the markdown format. For each item, include:
#                 - Date in **YYYY-MM-DD** format in Pakistan Standard Timezone
#                 - Concise sentences summary from latest news
#                 - Sort news datewise (latest first)
#                 - Source URL as link
#                 Use Format:
#                 ### [Date]
#                 - [Summary](URL)"""),
#             ("user", "Articles:\n{articles}")
#         ])
        
#         articles_str = "\n\n".join([
#             f"Content: {item.get('content', '')} \nURL: {item.get('url', '')} \nDate: {item.get('published_date', '')}"
#             for item in news_items
#         ])
        
#         response = self.llm.invoke(prompt_template.format(articles=articles_str))
#         state["summary"] = response.content
#         self.state["summary"] = state["summary"]
#         return self.state
    
#     def save_result(self, state):
#         frequency = self.state['frequency']
#         summary = self.state["summary"]
#         filename = f"./AINews/{frequency}_summary.md"
#         with open(filename, "w") as f:
#             f.write(f"# {frequency.capitalize()} AI News Summary\n\n")
#             f.write(summary)
#         self.state["filename"] = filename
#         return self.state


from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate
from src.Langgraph_Agentic_AI.state.state import State


class AI_News_Node:
    def __init__(self, llm):
        """
        Initialize the AI News Node with Tavily and LLM
        """
        self.tavily = TavilyClient()
        self.llm = llm

        # Internal storage for node processing
        self.state = {}

    def fetch_news(self, state: State) -> State:
        """
        Fetch AI news based on the messages in state.
        """
        raw_text = ""
        for msg in state.get("messages", []):
            if hasattr(msg, "content") and msg.content:
                raw_text = msg.content.lower().strip()
                break

        allowed_frequencies = ["daily", "weekly", "monthly", "year"]
        frequency = next(
            (f for f in allowed_frequencies if f in raw_text),
            "daily"
        )

        days_map = {"daily": 1, "weekly": 7, "monthly": 30, "year": 366}

        # Save frequency internally
        self.state["frequency"] = frequency

        # Fetch news from Tavily
        response = self.tavily.search(
            query="Top Artificial Intelligence (AI) technology news Pakistan and globally",
            topic="news",
            include_answer="advanced",
            max_results=15,
            days=days_map[frequency],
        )

        # Save news in internal state and LangGraph state
        self.state["news_data"] = response.get("results", [])
        state["news_data"] = self.state["news_data"]  # type: ignore

        return state

    def summarize_news(self, state: State) -> State:
        """
        Summarize fetched news using the LLM.
        """
        news_items = self.state.get("news_data", [])

        prompt_template = ChatPromptTemplate([
            ("system", """Summarize AI news into markdown. For each item include:
            - Date in **YYYY-MM-DD** format (Pakistan timezone)
            - Concise summary
            - Sort news latest first
            - Source URL as link
            Format:
            ### [Date]
            - [Summary](URL)
            """),
            ("user", "Articles:\n{articles}")
        ])

        articles_str = "\n\n".join([
            f"Content: {item.get('content', '')}\nURL: {item.get('url', '')}\nDate: {item.get('published_date', '')}"
            for item in news_items
        ])

        response = self.llm.invoke(prompt_template.format(articles=articles_str))

        # Save summary internally and in LangGraph state
        self.state["summary"] = response.content
        state["summary"] = self.state["summary"]  # type: ignore

        return state

    def save_result(self, state: State) -> State:
        """
        Save the summarized news to a markdown file.
        """
        frequency = self.state.get("frequency", "daily")
        summary = self.state.get("summary", "")

        filename = f"./AINews/{frequency}_summary.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {frequency.capitalize()} AI News Summary\n\n")
            f.write(summary)

        # Save filename internally and in LangGraph state
        self.state["filename"] = filename
        state["filename"] = filename  # type: ignore

        return state
