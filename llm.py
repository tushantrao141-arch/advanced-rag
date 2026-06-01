from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(find_dotenv())


class LLM:

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    def generate(self, query: str, context: str) -> str:
        template = """
You are an assistant that answers questions using only the given context.
If the context does not contain enough information to answer, say so clearly.

Context:
{context}

Question:
{query}

Answer:
"""
        llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature
        )

        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=template
        )

        chain = prompt | llm | StrOutputParser()

        return chain.invoke({
            "context": context,
            "query": query
        })