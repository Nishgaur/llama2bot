from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()


# Ensure that the environment variables are set correctly
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key is None:
    raise ValueError("LANGCHAIN_API_KEY environment variable is not set.")

# Set the environment variables

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = str(langchain_api_key)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Respond to the prompts"),
    ("user", "Question: {question}")
])

# Streamlit framework setup
st.title("LangChain Demo with Open sOrce Ollama")
input_text = st.text_input("Search")

# Ollama llama2 setup
llm = Ollama(model="llama2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser  # Combining all the tasks

# Handling input and displaying the output
if input_text:
    st.write(chain.invoke({'question': input_text}))
