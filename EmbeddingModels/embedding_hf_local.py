from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
load_dotenv()
# if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
#     raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not found")
# Generate embedding for a single query
vector = embedding.embed_query("Hello, how are you?")
print(str(vector))
# Generate embeddings for multiple documents
document_vector = embedding.embed_documents(["Hello, how are you?","I am fine."])
print(str(document_vector)) 
