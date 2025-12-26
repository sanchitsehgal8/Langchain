from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()

print("TOKEN:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))


if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not found")

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=128,
    temperature=0.7
)

chat = ChatHuggingFace(llm=llm)

response = chat.invoke("Hello, how are you?")
print(response.content)
