import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Use a currently supported embedding model by default, with env override.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
