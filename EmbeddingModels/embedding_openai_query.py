from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found")  
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",dimensions=32)
# Generate embedding for a single query
vector = embeddings.embed_query("Hello, how are you?")
print(str(vector))
# Generate embeddings for multiple documents
document_vector = embeddings.embed_documents(["Hello, how are you?","I am fine."])
print(str(document_vector))