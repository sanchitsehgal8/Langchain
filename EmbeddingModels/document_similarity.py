from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
#     raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not found")      
# Generate embeddings for multiple documents
documents = ["Hello, how are you?","I am fine.","What is your name?","My name is Langchain."]
query = "i am making ai agents using this technology?"
document_vectors = embedding.embed_documents(documents)
query_vector = embedding.embed_query(query)
# Compute cosine similarity between the first document and all others
similarities = cosine_similarity(
    [query_vector], 
    document_vectors
)[0]
print("Similarities:", similarities)

print("Most similar document:", documents[np.argmax(similarities)])