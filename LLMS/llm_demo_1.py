from dotenv import load_dotenv
import os

load_dotenv()

print("KEY FOUND:", os.getenv("OPENAI_API_KEY"))
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI

llm = OpenAI(model = "gpt-3.5-turbo")
result = llm.invoke("Explain the theory of relativity in simple terms.")
print(result)