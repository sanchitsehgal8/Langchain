from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  
import os
load_dotenv()
model = ChatOpenAI(model='gpt-4')
os.environ.get("OPENAI_API_KEY")
result = model.invoke("Hello, how are you?")