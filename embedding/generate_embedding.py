import json
import os
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Initialize OpenAI Embeddings
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
    )
index = pc.Index(INDEX_NAME)

# Load data and generate embeddings
with open("data/da_live_docs.json", "r") as f:
    docs = json.load(f)

for doc in docs:
    content = doc["content"]
    metadata = {
        "url": doc["url"],
        "title": doc["title"],
        "content": content,
        "images": doc.get("images", []),
        "youtube_links": doc.get("youtube_links", [])
    }
    
    # Generate embeddings
    vector = embeddings.embed_query(content)
    
    # Upsert into Pinecone
    index.upsert(vectors=[(doc["url"], vector, metadata)])

print(f"Uploaded {len(docs)} documents to Pinecone.")
