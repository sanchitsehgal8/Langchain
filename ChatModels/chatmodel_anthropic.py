from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv  
import os   
load_dotenv()
model = ChatAnthropic(model='claude-2')
os.environ.get("ANTHROPIC_API_KEY")
result = model.invoke("Hello, how are you?")
